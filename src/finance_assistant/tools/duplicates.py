"""R7 — duplicate ledger candidate detection by economic fingerprint.

`txn_id` and `doc_ref` are unique by construction, so a repeated-`doc_ref` search
finds nothing: duplicate detection here works off an economic fingerprint instead,
at three confidence tiers (HIGH/MEDIUM/LOW), with the matching window and amount
tolerance as configurable parameters rather than constants buried in the logic.

Framing requirement (verbatim from docs/PROMPT_MAESTRO.md): the ledger only
supports detecting *probable duplicate entries*. Proving an actual *double
payment* needs treasury evidence (payment file, bank statement, AP settlement
status) that is absent from this dataset — this module never claims otherwise.

R7 also requires checking whether a candidate was reversed by a credit memo (a
negative-amount row whose memo references the original doc_ref): a reversed
entry is not a double payment. That evidence is attached to every candidate
(`is_reversed`, `reversal_a`/`reversal_b`) rather than used to silently drop the
candidate — the same "report, don't hide" posture the rest of the tool layer
uses for FX gaps, unmapped accounts, and duplicate budget keys.
"""

import re
from dataclasses import dataclass

import pandas as pd

from finance_assistant.tools.vendors import AliasClusterDetectionResult

DEFAULT_DUPLICATE_WINDOW_DAYS = 7
DEFAULT_DUPLICATE_AMOUNT_TOLERANCE = 0.01  # cents-level float tolerance for "same amount"

_REQUIRED_COLUMNS = [
    "txn_id",
    "entity",
    "vendor_id",
    "currency",
    "amount",
    "memo",
    "accrual_date",
    "posting_date",
    "doc_ref",
]

_MEMO_PUNCTUATION_PATTERN = re.compile(r"[^\w\s]")
_MEMO_WHITESPACE_PATTERN = re.compile(r"\s+")


def normalize_memo_text(memo: str) -> str:
    normalized = memo.lower()
    normalized = _MEMO_PUNCTUATION_PATTERN.sub(" ", normalized)
    normalized = _MEMO_WHITESPACE_PATTERN.sub(" ", normalized).strip()
    return normalized


@dataclass(frozen=True)
class DuplicateDetectionRules:
    window_days: int = DEFAULT_DUPLICATE_WINDOW_DAYS
    amount_tolerance: float = DEFAULT_DUPLICATE_AMOUNT_TOLERANCE


@dataclass(frozen=True)
class ReversalEvidence:
    reversing_txn_id: str
    reversing_doc_ref: str
    reversing_amount: float


@dataclass(frozen=True)
class DuplicateCandidate:
    confidence: str  # "HIGH" | "MEDIUM" | "LOW"
    txn_id_a: str
    txn_id_b: str
    amount: float
    currency: str
    fingerprint: dict
    reversal_a: ReversalEvidence | None = None
    reversal_b: ReversalEvidence | None = None

    @property
    def is_reversed(self) -> bool:
        return self.reversal_a is not None or self.reversal_b is not None


DUPLICATE_DETECTION_LIMITATION = (
    "Detects probable duplicate ledger entries by economic fingerprint, not proven "
    "double payments -- proving a double payment needs treasury evidence (payment "
    "file, bank statement, AP settlement status) absent from this dataset. "
    "HIGH/MEDIUM tiers require vendor_id; rows without one are not evaluated. LOW "
    "tier requires the candidate's vendor to appear in a detect_alias_clusters() "
    "cluster -- supply alias_clusters=None to skip it entirely. Reversed candidates "
    "are annotated (is_reversed), never removed."
)


@dataclass(frozen=True)
class DuplicateDetectionResult:
    candidates: list[DuplicateCandidate]
    rules: DuplicateDetectionRules
    limitation: str = DUPLICATE_DETECTION_LIMITATION


def _find_reversals(rows: pd.DataFrame) -> dict[str, ReversalEvidence]:
    """doc_ref -> evidence that a negative-amount row's memo references it.

    Iterates only over amount < 0 rows (a small subset of the ledger) and checks
    which known doc_ref appears as a substring of each one's memo. Deliberately
    not locked to a specific reversal phrase (e.g. "reversal of") so it stays
    generic across differently-worded datasets.
    """
    negative = rows.loc[rows["amount"] < 0]
    doc_refs = [d for d in rows["doc_ref"].dropna().unique() if d]

    reversals: dict[str, ReversalEvidence] = {}
    for record in negative.to_dict("records"):
        memo = record.get("memo")
        if not isinstance(memo, str):
            continue
        reversing_doc_ref = record.get("doc_ref")
        for doc_ref in doc_refs:
            if doc_ref != reversing_doc_ref and doc_ref in memo:
                reversals[doc_ref] = ReversalEvidence(
                    reversing_txn_id=record["txn_id"],
                    reversing_doc_ref=reversing_doc_ref,
                    reversing_amount=float(record["amount"]),
                )
    return reversals


def _stringify(value):
    if isinstance(value, pd.Timestamp):
        return value.date().isoformat()
    return value


def _pairs_within_window(
    working: pd.DataFrame, on: list[str], window_days: int | None, exclude_pairs: set[frozenset]
) -> pd.DataFrame:
    if working.empty:
        return working.iloc[0:0]

    merged = working.merge(working, on=on, suffixes=("_a", "_b"))
    merged = merged.loc[merged["txn_id_a"] < merged["txn_id_b"]]

    if window_days is not None:
        window = pd.Timedelta(days=window_days)
        diff = (merged["posting_date_b"] - merged["posting_date_a"]).abs()
        merged = merged.loc[diff <= window]

    if exclude_pairs:
        pair_keys = merged.apply(lambda r: frozenset({r["txn_id_a"], r["txn_id_b"]}), axis=1)
        merged = merged.loc[~pair_keys.isin(exclude_pairs)]

    return merged


def detect_duplicate_candidates(
    rows: pd.DataFrame,
    rules: DuplicateDetectionRules | None = None,
    alias_clusters: AliasClusterDetectionResult | None = None,
) -> DuplicateDetectionResult:
    missing_columns = [c for c in _REQUIRED_COLUMNS if c not in rows.columns]
    if missing_columns:
        raise ValueError(f"rows is missing required column(s): {', '.join(missing_columns)}")
    for date_column in ("accrual_date", "posting_date"):
        if not pd.api.types.is_datetime64_any_dtype(rows[date_column]):
            raise ValueError(f"'{date_column}' is not a datetime column")

    rules = rules or DuplicateDetectionRules()
    reversals = _find_reversals(rows)

    working = rows.copy()
    working["_normalized_memo"] = working["memo"].map(
        lambda m: normalize_memo_text(m) if isinstance(m, str) else m
    )
    has_vendor = working["vendor_id"].notna()

    candidates: list[DuplicateCandidate] = []
    seen_pairs: set[frozenset] = set()

    # HIGH: same entity + vendor_id + currency + memo + accrual_date, amount
    # within tolerance, distinct txn_id.
    high_pool = working.loc[has_vendor]
    high_pairs = _pairs_within_window(
        high_pool, on=["entity", "vendor_id", "currency", "memo", "accrual_date"], window_days=None, exclude_pairs=set()
    )
    if not high_pairs.empty:
        high_pairs = high_pairs.loc[(high_pairs["amount_a"] - high_pairs["amount_b"]).abs() <= rules.amount_tolerance]
    for record in high_pairs.to_dict("records"):
        pair_key = frozenset({record["txn_id_a"], record["txn_id_b"]})
        seen_pairs.add(pair_key)
        candidates.append(
            DuplicateCandidate(
                confidence="HIGH",
                txn_id_a=record["txn_id_a"],
                txn_id_b=record["txn_id_b"],
                amount=float(record["amount_a"]),
                currency=record["currency"],
                fingerprint={
                    "entity": record["entity"],
                    "vendor_id": record["vendor_id"],
                    "currency": record["currency"],
                    "memo": record["memo"],
                    "accrual_date": _stringify(record["accrual_date"]),
                },
                reversal_a=reversals.get(record["doc_ref_a"]),
                reversal_b=reversals.get(record["doc_ref_b"]),
            )
        )

    # MEDIUM: same vendor_id + currency + amount + normalized memo,
    # posting_date within the configured window.
    medium_pool = working.loc[has_vendor]
    medium_pairs = _pairs_within_window(
        medium_pool,
        on=["vendor_id", "currency", "amount", "_normalized_memo"],
        window_days=rules.window_days,
        exclude_pairs=seen_pairs,
    )
    for record in medium_pairs.to_dict("records"):
        pair_key = frozenset({record["txn_id_a"], record["txn_id_b"]})
        seen_pairs.add(pair_key)
        candidates.append(
            DuplicateCandidate(
                confidence="MEDIUM",
                txn_id_a=record["txn_id_a"],
                txn_id_b=record["txn_id_b"],
                amount=float(record["amount"]),
                currency=record["currency"],
                fingerprint={
                    "vendor_id": record["vendor_id"],
                    "currency": record["currency"],
                    "amount": float(record["amount"]),
                    "normalized_memo": record["_normalized_memo"],
                },
                reversal_a=reversals.get(record["doc_ref_a"]),
                reversal_b=reversals.get(record["doc_ref_b"]),
            )
        )

    # LOW: same amount + currency + window, different vendor_id but both in the
    # same candidate alias cluster.
    if alias_clusters is not None:
        vendor_to_cluster = {
            vendor_id: cluster.normalized_name
            for cluster in alias_clusters.clusters
            for vendor_id in cluster.vendor_ids
        }
        low_pool = working.copy()
        low_pool["_cluster"] = low_pool["vendor_id"].map(vendor_to_cluster)
        low_pool = low_pool.loc[low_pool["_cluster"].notna()]
        low_pairs = _pairs_within_window(
            low_pool, on=["_cluster", "currency", "amount"], window_days=rules.window_days, exclude_pairs=seen_pairs
        )
        if not low_pairs.empty:
            low_pairs = low_pairs.loc[low_pairs["vendor_id_a"] != low_pairs["vendor_id_b"]]
        for record in low_pairs.to_dict("records"):
            candidates.append(
                DuplicateCandidate(
                    confidence="LOW",
                    txn_id_a=record["txn_id_a"],
                    txn_id_b=record["txn_id_b"],
                    amount=float(record["amount"]),
                    currency=record["currency"],
                    fingerprint={
                        "currency": record["currency"],
                        "amount": float(record["amount"]),
                        "alias_cluster": record["_cluster"],
                        "vendor_id_a": record["vendor_id_a"],
                        "vendor_id_b": record["vendor_id_b"],
                    },
                    reversal_a=reversals.get(record["doc_ref_a"]),
                    reversal_b=reversals.get(record["doc_ref_b"]),
                )
            )

    return DuplicateDetectionResult(candidates=candidates, rules=rules)
