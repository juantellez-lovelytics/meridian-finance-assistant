# ARCHITECTURE

**The system is agentic at the interpretation boundary and deterministic at
the financial-computation boundary.**

> "The LLM understands the question. Deterministic tools establish the
> facts. The orchestrator decides whether those facts are sufficient to
> answer." — `docs/PROMPT_MAESTRO.md`

## Las tools y por qué separadas así

Cada tool en `tools/` implementa exactamente una regla de
`docs/PROMPT_MAESTRO.md` (R1-R8), como función pura de dataclasses:

| Tool | Regla | Responsabilidad |
|---|---|---|
| `query_ledger` | R1 | Filtra por fecha financiera (default `accrual_date`) |
| `resolve_account_hierarchy` | R3 | Join temporal GL↔COA por vigencia |
| `convert_to_usd`/`aggregate_usd(_by)` | R2 | FX por (mes, moneda); cobertura explícita |
| `query_budget` | R5 | Detecta claves repetidas, nunca `drop_duplicates()` ciego |
| `normalize_reporting_cost_centre` | R4 | Transición de cost-centre, preserva el origen |
| `search_documents` | R8 | Keyword/sección, sin vector DB |
| `vendor_lookup`/`detect_alias_clusters` | R6 | Left join; detecta alias sin aplicarlos |
| `detect_duplicate_candidates` | R7 | Fingerprint económico + reversals |
| `evaluate_te_policy` | — | Reglas parametrizadas desde `policy_rules.yaml` |

Cada una tiene su propio test de caja negra. Un `workflow/` compone varias
tools en un plan fijo; el modelo nunca llama a una tool directamente.

## Qué decide el modelo

Dos responsabilidades acotadas, sin aritmética ni evidencia: **Question
Interpreter** (una llamada, `IntentRequest` tipado; confianza baja →
`NEEDS_CLARIFICATION`) y **Answer Renderer** (opcional, prosa desde un
`EvidenceBundle` ya decidido — no implementado). El Interpreter está
verificado contra un proveedor real; sin credencial, `answer_question` cae
a `interpret_with_keywords` y lo declara en `assumptions`.
`orchestration.plans` resuelve el intent a un workflow y completa los
parámetros que el modelo no puede conocer.

## Qué es determinista

Todo lo demás: joins temporales, FX, degradación del Evidence Gate,
cobertura, trace, renderer. Ningún número de un `EvidenceBundle` pasó por
un LLM.

## Por qué la frontera está ahí

El LLM se acota a interpretar lenguaje natural y, opcionalmente, a enunciar
en prosa un resultado ya fijado. Todo lo que toca números o suficiencia de
evidencia es Python puro: testeable, reproducible byte a byte, auditable
sin invocar un LLM.

## Dónde deliberadamente NO se usó un agente

- `workflows/` son planes fijos, no un loop de agente eligiendo la próxima
  tool.
- `search_documents` no tiene default de `filenames` — nadie busca los 4
  documentos "por si acaso".
- `evaluate_te_policy`/`detect_duplicate_candidates` devuelven candidatos
  tipados, nunca un veredicto.

## Cómo se manejan datos faltantes y rechazos

`evidence/gate.py` degrada de forma determinista
(`REFUSED > NEEDS_CLARIFICATION > PARTIAL > ANSWER`); el modelo no
participa. FX faltante nunca se interpola: se propaga como `MissingFXRate`.
Cobertura menor al 100% nunca produce `ANSWER`. Ejemplo real
(`traces/samples/*consolidated_spend*.json`, Q3): 1.201/1.348 filas
convertibles por una tasa EUR faltante — `REFUSED`, `result: null`, pero
`calculations` entrega el total computable y el desglose (ver
"Discrepancia argumentada").

## Cómo se rastrea la evidencia

`EvidenceBundle` acumula `ToolCall`/`CalcStep` vía `ToolTrace`
(`workflows/_shared.py`) — un único punto de instrumentación.
`evidence/trace.py::build_trace` proyecta el bundle a `steps[]`;
`write_trace` serializa a `traces/<run_id>.json`. `summarize.py` distingue
resumen de resultado (completo) de resumen de argumento (una línea), para
no duplicar lo que un paso anterior ya mostró.

## Cómo se acotan loops y coste

Tres ceilings en `.env.example`, aplicados por
`orchestrator._check_ceilings` tras la llamada al intérprete:
`LLM_MAX_CALLS_PER_QUESTION` (pasos), `LLM_MAX_TOKENS_PER_QUESTION`
(prompt+completion acumulados), `LLM_MAX_COST_USD_PER_QUESTION` (dinero).
Exceder cualquiera da `status=ERROR` con el motivo en `warnings`; el
workflow nunca corre. Precio desconocido → `"unknown"` en el costo, nunca
cero, declarado en `assumptions`.

## Por qué se rechazó ReAct abierto

Las ocho preguntas son un conjunto cerrado de rutas. Un plan fijo da la
misma cobertura que ReAct con selección abierta, sin sus riesgos:
secuencia incorrecta en tiempo real, scratchpad/auto-corrección,
coste/latencia no deterministas. A cambio, reproducibilidad total — cada
plan es un test unitario, no una política aprendida.

## Discrepancia argumentada

`docs/PROMPT_MAESTRO.md` invita explícitamente a discrepar de alguno de sus
propios principios. El que discuto es:

> "A plausible wrong answer costs more than a refusal."

Comparto la dirección — preferir un rechazo transparente a una respuesta
plausible pero no sustentada es correcto — pero la formulación es
incompleta, y este proyecto lo demuestra. La dicotomía que plantea opone
"respuesta plausible-pero-errónea" contra "rechazo", como si fueran las
únicas dos categorías relevantes. No lo son: la categoría que de verdad
importa es una tercera — el rechazo que además entrega lo que sí se pudo
establecer, frente al rechazo que no entrega nada.

La evidencia está en el propio repo. En Q3 (arriba) falta la tasa EUR de
2024-09, así que el total consolidado exacto no es defendible y el sistema
lo rechaza — correcto. Pero un rechazo seco habría tirado a la basura
información perfectamente sólida: 1.201 de 1.348 filas eran convertibles,
los componentes por entidad estaban calculados, y el importe local no
convertible estaba cuantificado con precisión. El bundle devuelve
`result: null` — el total exacto genuinamente no existe — y a la vez
entrega esos tres bloques en `calculations`. El analista se lleva el 89%
del cuadro y sabe exactamente qué le falta y por qué.

De ahí sale una consecuencia de diseño que el principio original no
menciona: **el estado del bundle y el contenido del bundle son ejes
independientes.** Un bundle `REFUSED` puede llevar más información útil que
un `ANSWER` pobre. Si se colapsan los dos ejes — si "rechazar" se entiende
como "no decir nada" — el sistema aprende a callarse precisamente cuando
debería estar acotando lo que sabe. La prueba de que esto no es solo
retórica está en `evals/questions.yaml`: el caso de Q3 no solo asevera que
`status == REFUSED`, también asevera que `calculations` trae los
componentes computables. Una regresión que dejara el rechazo mudo pasaría
el primer assert y fallaría el segundo — el eval está diseñado para cazar
exactamente el error de colapsar los dos ejes.

Reformulación que propondría en su lugar:

> "An unsupported answer costs more than a refusal — and a refusal that
> discards what could be established costs more than one that reports it."
