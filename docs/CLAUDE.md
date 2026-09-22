# Meridian Finance Analyst Assistant

Sistema de análisis financiero para una empresa financiera de ejemplo. La corrección y la
evidencia importan más que la autonomía del agente.

Principio rector:
"The LLM understands the question. Deterministic tools establish the facts.
The orchestrator decides whether those facts are sufficient to answer."

Ver docs/PROMPT_MAESTRO.md para las reglas completas (R1-R8).

## Reglas inviolables
- Toda aritmética en Python determinista. El LLM nunca suma, convierte moneda,
  rankea ni compara contra umbrales.
- Ninguna afirmación sin fuente trazable.
- Nada de `run_anything`, SQL generado por modelo, ni `exec()`.
- Nunca hardcodear valores de este dataset. Las anomalías se DETECTAN.
- `accrual_date` es la fecha financiera por defecto, nunca `posting_date`.
- FX faltante: nunca interpolar. Error estructurado con filas afectadas.
- Toda agregación en USD devuelve (valor, cobertura). Cobertura < 100% => nunca ANSWER.
- Join de chart of accounts SIEMPRE temporal (valid_from/valid_to).
- Prohibido `drop_duplicates()` ciego en budget o chart of accounts.
- Vendors: nunca fusionar por similitud de nombre. Detectar clusters, no aplicarlos.
- Left join de vendors: hay gasto legítimo sin vendor_id.

## Comandos
- Tests: `pytest -q`
- Evals: `python -m evals.run_evals`
- UI: `streamlit run src/finance_assistant/ui/app.py`

## Entorno
- Windows, PowerShell 5.1. NO usar `&&` para encadenar comandos.
- Un comando por línea.

## Convenciones
- Correr `pytest` antes de cada commit.
- Un commit por unidad lógica: `feat:` / `test:` / `docs:` / `chore:`.
- No modificar nada dentro de `data/`.