"""R6 — vendor left-join and deterministic alias-cluster detection.

`vendor_lookup` always left-joins: expense without a vendor_id (payroll,
bank fees, ...) is legitimate spend, not a data-quality defect, and must
survive the join. An inner join would silently drop it from any
vendor-based total. Rows with a vendor_id that isn't in the master are
kept too, but flagged separately — that *is* a data-quality signal.

`detect_alias_clusters` proposes candidate name clusters via a purely
mechanical normalization: lowercase, strip punctuation, drop a fixed list
of common corporate suffixes, collapse whitespace. It never merges
vendor_ids and there is no `canonical_vendor_id` anywhere in this module
— fuzzy matching, embeddings, and LLM judgment are all out of scope by
design (R6). Callers decide whether a candidate cluster would change a
ranking's composition enough to matter; this module only detects.
"""

import re
from dataclasses import dataclass

import pandas as pd

# Common corporate-suffix tokens stripped during alias normalization. A
# generic algorithm parameter (not read from a business document, unlike
# R4's cost-centre transition table): extending it to a new jurisdiction
# is a config change here, never an inference.
CORPORATE_SUFFIXES = [
    "incorporated",
    "corporation",
    "limited",
    "company",
    "inc",
    "llc",
    "llp",
    "ltd",
    "corp",
    "co",
    "gmbh",
    "bv",
    "b v",
    "nv",
    "plc",
    "sa",
    "srl",
    "ag",
]

_SUFFIX_PATTERN = re.compile(r"\b(" + "|".join(re.escape(s) for s in CORPORATE_SUFFIXES) + r")\b")
_PUNCTUATION_PATTERN = re.compile(r"[^\w\s]")
_WHITESPACE_PATTERN = re.compile(r"\s+")


def normalize_vendor_name(name: str) -> str:
    normalized = name.lower()
    normalized = _PUNCTUATION_PATTERN.sub(" ", normalized)
    normalized = _SUFFIX_PATTERN.sub(" ", normalized)
    normalized = _WHITESPACE_PATTERN.sub(" ", normalized).strip()
    return normalized


@dataclass(frozen=True)
class VendorLookupResult:
    rows: pd.DataFrame
    total_rows: int
    matched_rows: int  # vendor_id present and found in the master
    no_vendor_id_rows: int  # vendor_id null -> legitimate vendor-less spend
    unmatched_vendor_id_rows: int  # vendor_id present but absent from the master


def vendor_lookup(rows: pd.DataFrame, vendors: pd.DataFrame) -> VendorLookupResult:
    if "vendor_id" not in rows.columns:
        raise ValueError("rows must have a 'vendor_id' column")
    required_vendor_columns = ["vendor_id", "vendor_name", "category", "country"]
    missing_vendor_columns = [c for c in required_vendor_columns if c not in vendors.columns]
    if missing_vendor_columns:
        raise ValueError(f"vendors is missing required column(s): {', '.join(missing_vendor_columns)}")

    merged = rows.merge(vendors, on="vendor_id", how="left", suffixes=("", "_vendor"))

    has_vendor_id = rows["vendor_id"].notna().to_numpy()
    is_matched = merged["vendor_name"].notna().to_numpy()

    return VendorLookupResult(
        rows=merged,
        total_rows=len(rows),
        matched_rows=int((has_vendor_id & is_matched).sum()),
        no_vendor_id_rows=int((~has_vendor_id).sum()),
        unmatched_vendor_id_rows=int((has_vendor_id & ~is_matched).sum()),
    )


@dataclass(frozen=True)
class AliasCluster:
    normalized_name: str
    vendor_ids: list[str]
    vendor_names: list[str]


# Attached to every detect_alias_clusters() result, not just documented here,
# so it travels with the data into the evidence/trace layer instead of
# staying implicit in a docstring. Mechanical normalization is deliberately
# blind to abbreviations (a shortened form of the same word) and
# translations (the same name in another language) -- neither collapses to
# the same normalized string, so real alias risk is understated by the
# clusters actually returned. Closing that gap needs an authoritative
# canonical vendor mapping, which does not exist in this dataset; it must
# not be approximated here via fuzzy matching, embeddings, or LLM judgment.
ALIAS_CLUSTER_LIMITATION = (
    "Alias detection here is purely mechanical (lowercase, punctuation "
    "stripped, common corporate suffixes removed, whitespace collapsed). "
    "It does not detect abbreviations or translations of the same vendor "
    "name, so the actual number of alias vendors is likely higher than the "
    "clusters returned. Resolving those cases requires an authoritative "
    "canonical vendor mapping, which is absent from this dataset -- it must "
    "not be inferred via fuzzy matching or LLM judgment."
)


@dataclass(frozen=True)
class AliasClusterDetectionResult:
    clusters: list[AliasCluster]
    limitation: str = ALIAS_CLUSTER_LIMITATION


def detect_alias_clusters(vendors: pd.DataFrame) -> AliasClusterDetectionResult:
    required_columns = ["vendor_id", "vendor_name"]
    missing_columns = [c for c in required_columns if c not in vendors.columns]
    if missing_columns:
        raise ValueError(f"vendors is missing required column(s): {', '.join(missing_columns)}")

    working = vendors.copy()
    working["_normalized_name"] = working["vendor_name"].map(normalize_vendor_name)

    clusters = []
    for normalized_name, group in working.groupby("_normalized_name"):
        if len(group) < 2:
            continue
        clusters.append(
            AliasCluster(
                normalized_name=normalized_name,
                vendor_ids=group["vendor_id"].tolist(),
                vendor_names=group["vendor_name"].tolist(),
            )
        )
    clusters.sort(key=lambda cluster: cluster.normalized_name)
    return AliasClusterDetectionResult(clusters=clusters)
