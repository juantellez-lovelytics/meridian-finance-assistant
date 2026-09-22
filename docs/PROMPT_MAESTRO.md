# PROMPT MAESTRO — Meridian Finance Analyst Assistant

> Pegar íntegro como primer mensaje en Claude Code (o el agente de codificación que uses).
> Está diseñado para **evitar sobre-ingeniería** y para forzar las decisiones defendibles.

---

## ROL

Eres el ingeniero principal responsable de construir, para una empresa financiera de ejemplo (ficticia), un asistente de análisis financiero listo para producción. Quien lo revise podrá **clonar el repo, correrlo con sus propias credenciales, y ejecutar tus tools contra un segundo dataset con las mismas columnas y números distintos**.

Esto **no** es un chatbot. Es un sistema de análisis financiero donde importan, en este orden: corrección, evidencia, incertidumbre explícita, comportamiento de rechazo, cálculo determinista y orquestación explicable. La autonomía del agente es lo *menos* importante.

Frase que gobierna todo el diseño:

> **The LLM understands the question. Deterministic tools establish the facts. The orchestrator decides whether those facts are sufficient to answer.**

---

## RESTRICCIONES DURAS (no negociables)

- Todo el aritmética ocurre en Python determinista. El LLM **nunca** suma, convierte moneda, calcula varianzas, rankea ni compara contra umbrales.
- Ninguna afirmación sin fuente trazable (archivo + filtro + fila/sección).
- Tools pequeñas y específicas. **Nada** de `run_anything`, ni SQL generado por el modelo, ni `exec()` de código del modelo.
- Todo loop y toda interacción con el modelo tiene techo explícito (pasos, tokens, coste).
- Un rechazo transparente es mejor que una respuesta plausible sin respaldo.
- **Nunca** hardcodear valores numéricos de este dataset en la lógica. Las anomalías se **detectan**, no se asumen.
- Nunca commitear `.env`, keys o tokens. `.gitignore` fuerte desde el primer commit.
- Sin auth, sin deploy, sin base de datos, sin vector store.
- No mutar los archivos fuente.

---

## DATOS DE ENTRADA

```
data/
  gl_transactions.csv      # 10.916 filas, 12 columnas
  chart_of_accounts.csv    # 34 filas, con valid_from/valid_to
  budget.csv               # 2.100 filas, todo en USD, solo 2024
  fx_rates.csv             # 71 filas
  vendors.csv              # 43 filas
data/documents/
  board_memo_2024_q2.md
  contract_kestrel.md
  contract_northgate_advisory.md
  travel_expense_policy.md
```

Los loaders validan columnas requeridas y tipos, y fallan con errores legibles.

---

## INVARIANTES Y REGLAS DE NEGOCIO

Cada regla se implementa de forma **genérica y data-driven**. Las características concretas del dataset actual se listan solo como *fixtures esperados en tests*, nunca como ramas de código.

### R1 — Base temporal

`accrual_date` es la fecha financiera por defecto. Constante configurable:

```python
DEFAULT_FINANCIAL_DATE_FIELD = "accrual_date"
```

Motivo: existen transacciones devengadas en un año y contabilizadas en el siguiente. Filtrar por `posting_date` produce periodos incorrectos. La base de fecha usada debe aparecer en el trace y en la respuesta cuando sea material.

*Fixture de test:* deben existir filas con `posting_date.year > accrual_date.year`; el filtro de FY2024 debe incluirlas. El sistema **reporta** cuántas encontró, no las asume.

### R2 — FX

- El GL está en moneda local por entidad; el budget está en USD.
- Join de FX: `(mes de la fecha financiera, moneda)`. Fórmula: `amount_usd = amount * rate_to_usd`.
- Si falta una tasa requerida: **nunca** interpolar, ni usar mes anterior/posterior, ni asumir, ni buscar en internet, ni ocultar las filas.
- Se emite un error estructurado:

```python
MissingFXRate(currency, period_month, affected_rows, affected_amount_local)
```

- **Regla crítica de agregación:** las filas no convertibles **no pueden desaparecer silenciosamente en un `sum()`**. Un `groupby().sum()` con NaN produce totales que parecen correctos y no lo son. Toda agregación en USD devuelve `(valor, cobertura)` donde cobertura = filas convertidas / filas seleccionadas, y si cobertura < 100% el estado nunca puede ser `ANSWER`.

*Fixture:* falta exactamente una combinación moneda/mes. Detectarla genéricamente comparando el producto cartesiano (meses presentes en GL × monedas presentes en GL) contra `fx_rates`.

### R3 — Chart of accounts temporal

El COA tiene `valid_from` / `valid_to`. Un mismo `account_code` puede tener padres distintos en periodos distintos.

Prohibido: `drop_duplicates("account_code")`, `set_index("account_code")`, o cualquier join que ignore vigencia.

Join obligatorio:

```
txn.account_code == coa.account_code
AND txn.<financial_date> >= coa.valid_from
AND txn.<financial_date> <= coa.valid_to
```

Manejar `9999-12-31`. Test: ninguna transacción debe quedar con 0 mapeos ni con >1 mapeo.

*Fixture:* al menos una cuenta cambia de padre a mitad de 2024. No hardcodear el código de esa cuenta.

### R4 — Cost centre reporting

El board memo define una transición de código de centro de coste efectiva a mitad de año, con el plan del año restated bajo el código nuevo y los comparativos históricos **sin** restatement.

- Nunca destruir el valor origen. Mantener dos campos: `source_cost_centre` y `reporting_cost_centre`.
- La normalización es una función explícita, aplicada **solo** cuando la comparabilidad lo exige (actual vs budget, o comparaciones que cruzan la fecha efectiva).
- Cada normalización aplicada queda registrada en el trace con referencia a la regla documental que la justifica.
- La regla de mapeo se lee del documento y se declara en configuración; no se infiere con el LLM.

### R5 — Budget

- Budget en USD; nunca comparar contra importes locales sin convertir.
- **Existen claves dimensionales repetidas** (misma entidad + centro + cuenta + mes con importes distintos). Prohibido `drop_duplicates()` ciego y prohibido elegir una fila al azar.
- El sistema debe: (a) detectar las claves repetidas, (b) reportarlas como diagnóstico de calidad con conteo e importe, (c) aplicar una regla de agregación **declarada y justificable**, (d) exponer esa decisión como *assumption* en la respuesta.
- La justificación por defecto es aditiva (las filas repetidas son componentes del plan restated, no versiones alternativas). El sistema debe **poder demostrarlo empíricamente**: implementa un chequeo de plausibilidad que compare la ratio actual/budget del centro afectado bajo ambas hipótesis contra la ratio mediana del resto de centros, y registra el resultado en el trace. Si ambas hipótesis fueran igual de plausibles, degradar a `PARTIAL` y exponer las dos cifras.
- Budget solo cubre un año. Cualquier pregunta de varianza fuera de ese rango es `REFUSED / NO_BUDGET_FOR_PERIOD`.

### R6 — Vendors

- `vendor_id` es el identificador autoritativo.
- El maestro contiene nombres que **parecen** alias del mismo proveedor (variantes de mayúsculas, sufijos societarios, traducciones).
- **Prohibido** fusionar automáticamente con fuzzy matching, embeddings o criterio del LLM. No existe `canonical_vendor_id` en los datos.
- Pero **sí es obligatorio medir el impacto**: implementa `detect_alias_clusters()` determinista (normalización de nombre: minúsculas, quitar puntuación y sufijos societarios comunes, colapsar espacios) que devuelve *candidatos* agrupados **sin** aplicarlos.
- Para rankings de proveedores, calcular el ranking en **ambas** bases (por `vendor_id` y por cluster candidato) y comparar. Si la fusión cambiaría la composición del top-N, el estado es `PARTIAL` y la respuesta presenta las dos vistas. Un ranking presentado como definitivo cuando la agrupación lo alteraría es una respuesta incorrecta.
- Filas sin `vendor_id` son legítimas (gasto sin proveedor, p.ej. nómina). Usar `left join`; un `inner join` las borraría silenciosamente. Reportarlas como categoría aparte, nunca omitirlas del total.

### R7 — Credit memos / reversals

El ledger contiene importes negativos que revierten documentos concretos. Se requiere:

- Distinguir importes brutos de netos y declarar cuál se está usando.
- Resolver la referencia del documento revertido cuando el memo la contenga.
- En detección de duplicados, comprobar si el candidato fue revertido; un asiento revertido no es un pago doble.

### R8 — Documentos

- Los documentos son fuentes de evidencia, no contexto de relleno.
- **Prohibido** meter los cuatro documentos en el contexto del LLM para cada pregunta.
- Búsqueda determinista por keyword/sección sobre cuatro Markdown pequeños. Nada de vector DB.
- Devuelve: `filename`, `section`, `snippet`, `evidence_id`.
- **Orden causal obligatorio:** los números establecen el hecho; el documento **explica** el hecho. Nunca al revés. Si un memo menciona una categoría de gasto, eso no prueba que esa categoría cause una desviación: la desviación se establece primero desde el ledger, por cuenta, y solo entonces se busca explicación documental. Si el documento y los números no coinciden exactamente, **decirlo**.

---

## MODELO DE EVIDENCIA

Toda workflow devuelve un `EvidenceBundle` antes de que se genere cualquier texto final. El renderer **solo** puede usar valores contenidos en ese bundle.

```python
class AnswerStatus(str, Enum):
    ANSWER = "answer"
    PARTIAL = "partial"
    REFUSED = "refused"
    NEEDS_CLARIFICATION = "needs_clarification"
    ERROR = "error"

class EvidenceBundle(BaseModel):
    status: AnswerStatus
    intent: Intent
    result: dict | None
    sources: list[SourceRef]
    assumptions: list[str]
    warnings: list[str]
    missing_evidence: list[MissingEvidence]
    coverage: Coverage          # selected_rows, computable_rows, computable_amount_pct
    calculations: list[CalcStep]
    tool_calls: list[ToolCall]
    refusal_reason: str | None
```

**Evidence Gate** — función determinista, no un prompt. Degrada el estado según reglas:

| Condición | Estado forzado |
|---|---|
| Falta una tasa FX requerida y afecta el total pedido | `PARTIAL` o `REFUSED` |
| Falta el denominador de un ratio | `REFUSED` |
| Cobertura de filas < 100% en un total consolidado | máximo `PARTIAL` |
| El periodo pedido es ambiguo y las lecturas dan resultados materialmente distintos | `NEEDS_CLARIFICATION` |
| Una decisión de agrupación no autoritativa cambiaría el resultado | máximo `PARTIAL` |
| No hay dato para el periodo pedido | `REFUSED` |

El modelo **no participa** en esta decisión.

---

## TOOLS (pequeñas, deterministas, testeables sin LLM)

```
query_ledger(date_start, date_end, date_field, entities, cost_centres, account_codes, vendor_ids)
resolve_account_hierarchy(rows)            # join temporal, R3
convert_to_usd(rows, target="USD")         # devuelve (rows, coverage, missing[]), R2
query_budget(period, cost_centres, accounts, entity)   # R5
normalize_reporting_cost_centre(rows, ctx)  # R4, preserva origen
search_documents(query, filename=None)      # R8
vendor_lookup(rows)                         # left join, R6
detect_alias_clusters(vendors)              # candidatos, no aplica, R6
detect_duplicate_candidates(rows, rules)    # niveles de confianza, R7
evaluate_te_policy(rows, policy)            # motor de reglas, ver abajo
```

Los **workflows** (uno por intención) son planes deterministas del orquestador, **no** tools expuestas al modelo.

---

## MOTOR DE POLÍTICA T&E

Reglas leídas de la policy, parametrizadas en un archivo de configuración (`policy_rules.yaml`), no incrustadas en código:

- Alojamiento: tope por noche distinto en ciudades tier-one vs resto; lista de ciudades tier-one desde el documento.
- Dietas: tope por día.
- Entretenimiento de cliente: por encima de un umbral requiere aprobación previa de VP; además requiere registro de asistentes.
- Cualquier gasto individual ≥ umbral requiere referencia de pre-aprobación registrada contra la transacción.
- Categorías no reembolsables (lista de keywords).

**Alcance:** la política es de *Travel & Entertainment*. Aplicar la regla de pre-aprobación a cuentas fuera de ese perímetro (alquiler, nómina, seguros) es un falso positivo masivo. El perímetro se deriva del COA temporal, y **la reclasificación de mitad de año lo mueve** — documenta explícitamente qué perímetro usas y por qué.

Cada evaluación devuelve por transacción y por regla:

```
CONFIRMED_RULE_MATCH   # el dato prueba el incumplimiento de la regla tal como está escrita
POTENTIAL_BREACH       # el dato sugiere incumplimiento pero falta un elemento
INSUFFICIENT_EVIDENCE  # el dato no permite evaluar la regla
NOT_A_BREACH
NOT_APPLICABLE
```

Calibración obligatoria:
- Importe ≥ umbral sin `approval_ref`, dentro del perímetro T&E → `CONFIRMED_RULE_MATCH` (la regla es verificable literalmente: la referencia debe estar registrada).
- Tarifa por noche por encima del tope → `POTENTIAL_BREACH`, porque el tope es *exclusive of local tax* y el GL no separa impuestos. Documentarlo.
- Clase business → `INSUFFICIENT_EVIDENCE` salvo que falte aprobación; la duración del vuelo no está en los datos y **no se infiere de la ciudad**.
- Entretenimiento de cliente: la aprobación es verificable; el registro de asistentes **no** lo es → resultado desdoblado por limbo de la regla.
- Si falta FX para llevar el importe al umbral en USD → `INSUFFICIENT_EVIDENCE` para esa transacción, nunca exclusión silenciosa.

Salida por hallazgo: `txn_id`, regla, valor observado, umbral, razón, fuente documental (archivo + sección), estado.

El sistema se presenta como **detector de candidatos**, nunca como auditor que determina culpabilidad.

---

## DETECCIÓN DE DUPLICADOS

`txn_id` y `doc_ref` son únicos por construcción: buscar `doc_ref` repetido no encuentra nada. La detección debe ser por **huella económica**.

```
HIGH:   misma entidad + vendor_id + moneda + importe + memo + fecha de devengo, txn_id distinto
MEDIUM: mismo vendor + moneda + importe + memo normalizado, posting_date dentro de ventana configurable
LOW:    mismo importe y moneda y ventana, vendor distinto pero en el mismo cluster de alias candidato
```

Ventanas y niveles son configurables, no constantes mágicas.

Framing obligatorio de la respuesta: el ledger permite detectar **asientos duplicados probables**; probar un **pago doble** requiere evidencia de tesorería (fichero de pagos, extracto bancario, estado de liquidación en AP) que no está en el dataset. Comprobar además si el candidato fue revertido por credit memo (R7).

---

## ORQUESTACIÓN

```
Question
  → Question Interpreter (LLM, salida estructurada Pydantic, 1 llamada)
  → IntentRequest
  → Plan Registry (determinista)
  → Tools
  → Evidence Gate (determinista)
  → Answer Renderer (LLM opcional, solo desde el EvidenceBundle)
  → Trace
```

`Intent` enum incluye `UNKNOWN`. `IntentRequest` incluye `confidence`; por debajo de un umbral → `NEEDS_CLARIFICATION`, no adivinar.

**Manejo de periodo ambiguo:** una pregunta como "Q2" o "Q3" sin año, sobre un dataset multi-anual, es ambigua. La regla: si las distintas lecturas producen resultados con estados o magnitudes materialmente distintos, devolver `NEEDS_CLARIFICATION` con las opciones; si no, asumir el año más reciente y **declarar la asunción**. Implementar esto como regla del gate, no como criterio del modelo.

Techos explícitos: máximo de llamadas al modelo por pregunta, máximo de tokens, coste estimado acumulado. Superarlo → `ERROR` con motivo, nunca continuar.

Sin credencial de LLM: los tools, tests y evals deterministas deben seguir corriendo. Proveer un intérprete de respaldo por keywords para poder demostrarlo.

---

## TRACE

Un JSON por ejecución en `traces/`:

```json
{
  "run_id": "...", "started_at": "...", "question": "...", "status": "...",
  "date_basis": "accrual_date",
  "steps": [{"step":2,"type":"tool","name":"convert_to_usd","arguments":{},"result_summary":{},"duration_ms":0}],
  "model_calls": [{"provider":"...","model":"...","prompt_tokens":0,"completion_tokens":0,"estimated_cost_usd":0,"latency_ms":0}],
  "final_evidence": {},
  "duration_ms": 0, "estimated_cost_usd": 0
}
```

Resúmenes legibles, no miles de filas crudas. Si el precio del modelo no se conoce, mostrar `"unknown"` en lugar de inventarlo. Commitear al menos tres traces representativos: una respuesta numérica correcta, un análisis de política, y un rechazo por evidencia insuficiente.

---

## LAS OCHO PREGUNTAS

Implementar como workflows. Comportamiento esperado de diseño:

| # | Pregunta | Diseño |
|---|---|---|
| 1 | Opex Q2 por cost centre | `ANSWER` (o `NEEDS_CLARIFICATION` de año). Declarar base de fecha y perímetro opex. |
| 2 | Travel 2024 vs 2023 | `PARTIAL`. Obligatorio presentar **base reportada** y **base comparable**, con el puente de reclasificación cuantificado. La variación sobre base reportada y sobre base comparable pueden tener **signo distinto**: si el sistema entrega solo una, la respuesta es engañosa. |
| 3 | Spend consolidado Q3 en USD | `REFUSED` para el total exacto por tasa FX faltante. Entregar componentes computables y el importe local no convertible. |
| 4 | Top 10 proveedores | `PARTIAL`. Ranking por `vendor_id` **y** por cluster candidato; declarar que la composición del top-10 depende de un mapeo no autoritativo. Test de materialidad del FX faltante sobre las posiciones frontera. |
| 5 | Peores centros vs budget en Q3 + driver | `PARTIAL`. Ranking de varianzas adversas; descomposición por cuenta del peor centro; advertir que los centros con FX incompleto aparecen artificialmente por debajo de plan. Buscar el memo **después** de establecer el driver numérico y señalar cualquier discrepancia entre la narrativa del memo y las cuentas que realmente mueven la cifra. |
| 6 | Transacciones que incumplen T&E | `ANSWER` como candidatos, desglosado por regla y por estado. Nunca como auditoría definitiva. |
| 7 | Coste de personal por FTE | `REFUSED`. El denominador (FTE) no existe en el ledger; el memo lo confirma. Nombrar el dataset ausente y citar la sección. |
| 8 | ¿Pagamos dos veces? | `ANSWER` como candidatos de asiento duplicado + declaración explícita de qué evidencia faltaría para afirmar pago doble. |

Los dos rechazos limpios son #3 y #7. Los demás requieren honestidad epistemológica, no negativa total.

---

## TESTS (pytest, sin LLM)

- Filtro de periodo usa `accrual_date`; transacción devengada antes de fin de año y contabilizada después queda en el periodo correcto.
- Join temporal del COA: cero filas sin mapeo, cero filas con mapeo ambiguo.
- Conversión FX: aritmética exacta contra valores calculados a mano.
- FX faltante produce fallo estructurado, no NaN silencioso.
- Una agregación con cobertura parcial **no** puede devolver estado `ANSWER`.
- Normalización de cost centre preserva el valor origen.
- Agregación de budget no descarta claves repetidas.
- Detección de duplicados encuentra los pares de huella idéntica.
- Umbrales de política se evalúan sobre importe en USD.
- Clusters de alias se detectan pero no se aplican automáticamente.
- Los loaders fallan con mensaje claro ante columna faltante.

**Los valores esperados se calculan de forma independiente, nunca invocando la misma función bajo test.** Tolerancia de coma flotante explícita.

---

## EVALS

```
evals/questions.yaml
evals/run_evals.py       →  python -m evals.run_evals   (exit code != 0 si falla)
```

Cada caso:

```yaml
- id: q_headcount
  question: "What's our headcount cost per FTE?"
  expected_intent: HEADCOUNT_COST_PER_FTE
  expected_status: REFUSED
  expected_reason: MISSING_FTE_DENOMINATOR
  required_sources: [board_memo_2024_q2.md]
  forbidden_claims: ["numeric cost per FTE"]
```

Separar evals deterministas (siempre corren) de evals de orquestación en vivo (requieren credencial). Los `forbidden_claims` se comprueban contra el texto renderizado: si aparece un número donde no debería, el eval falla.

---

## ESTRUCTURA

```
README.md  ARCHITECTURE.md  NOTES.md  .gitignore  .env.example  pyproject.toml
data/{csvs, documents/}
src/finance_assistant/{config,models}.py
  data/{loaders,validation}.py
  tools/{ledger,accounts,fx,budget,documents,vendors,duplicates,te_policy}.py
  workflows/{opex,travel,consolidated,vendors,variance,policy,headcount,duplicates}.py
  orchestration/{intents,interpreter,plans,orchestrator}.py
  evidence/{models,gate,trace}.py
  ui/app.py
tests/  evals/  traces/
```

No crear directorios sin propósito.

---

## UI

Streamlit. Claridad sobre decoración. Layout: pregunta → badge de estado → respuesta → evidencia clave (tabla/gráfico) → asunciones → advertencias → fuentes → "How I got this" (timeline expandible) → trace JSON crudo.

El badge de estado debe ser visualmente prominente. Un `REFUSED` bien presentado es una demo, no un fallo.

---

## DOCUMENTACIÓN

**README.md** — clonar y correr en minutos: qué hace, prerequisitos, instalación, configuración, proveedores soportados, cómo correr UI/CLI/tests/evals, estructura, pregunta de ejemplo, trace de ejemplo, screenshot, limitaciones conocidas.

**ARCHITECTURE.md** — una página. Debe responder: cuáles son las tools y por qué separadas; qué decide el modelo; qué es determinista; por qué la frontera está ahí; dónde deliberadamente **no** se usó un agente; cómo se manejan datos faltantes y rechazos; cómo se rastrea la evidencia; cómo se acotan loops y coste. Incluir la frase:

> "The system is agentic at the interpretation boundary and deterministic at the financial-computation boundary."

Explicar que se rechazó ReAct abierto porque las ocho rutas analíticas son previsibles y la corrección financiera se beneficia de menos autonomía. Esta especificación invita a discrepar de sus propios principios: si discrepas de alguno, argumentarlo aquí (es una invitación explícita y contestarla suma).

**NOTES.md** — primera persona, una página: qué herramientas de IA usaste y cómo, algo que la IA hizo particularmente bien, algo que hizo mal y **cómo lo detectaste**, una cosa en la que tuviste que corregirla repetidamente y **qué cambio arquitectónico hizo que dejara de pasar**, qué recortaste, tiempo aproximado, qué harías con dos días más.

No inventar fallos de IA que no ocurrieron. Escribirlo a medida que avanzas, no al final.

---

## COMMITS

Incrementales y con mensaje con significado. Nunca un commit gigante final.

```
chore: scaffold project and validate source data
feat: add deterministic ledger and fx tools
feat: support temporal chart of accounts
feat: add reporting cost-centre normalization
feat: implement budget variance workflows
feat: add evidence gate and refusal handling
feat: implement T&E policy rule engine
feat: detect duplicate ledger candidates
feat: add bounded question orchestrator
feat: add Streamlit trace viewer
test: add case eval suite
docs: document architecture and tradeoffs
docs: finalize AI usage notes and limitations
```

---

## LISTA DE PROHIBICIONES

- Aritmética en el modelo.
- Afirmación sin evidencia.
- Interpolación de FX.
- Join de COA ignorando vigencia.
- `drop_duplicates()` ciego en budget o COA.
- Fusión automática de proveedores.
- Ranking presentado como definitivo cuando una decisión no autoritativa lo alteraría.
- Filas no convertibles desapareciendo en una suma.
- `inner join` de proveedores que borre gasto sin proveedor.
- Afirmar pago doble desde evidencia de ledger.
- Calcular FTE sin dato de FTE.
- Conclusión de cumplimiento de política sin evidencia suficiente.
- SQL o Python generado por el modelo.
- Loop sin techo.
- Respuestas numéricas hardcodeadas.
- Secretos en el repo o en el historial de git.
- Corrección silenciosa de calidad de datos.

---

## MÉTODO DE TRABAJO

1. Antes de escribir código de aplicación: inspeccionar los esquemas, escribir en `NOTES.md` los invariantes y anomalías descubiertos, definir los modelos tipados, escribir los tests de tools.
2. Implementar la capa determinista completa y verde antes de tocar orquestación.
3. No implementar Streamlit hasta que el núcleo analítico sea correcto.
4. Después de cada workflow: correr tests, correr su eval, inspeccionar un trace.
5. Si en algún momento proponer un atajo mueve aritmética o validación de evidencia al LLM, **rechazar el atajo**.
6. Ante cualquier dato ausente preguntar: *¿puede establecerse esta afirmación desde la evidencia suministrada?* Si no, representar la limitación estructuralmente.

**Criterio de terminado:** entorno limpio instala; datos validan; tools con tests; ocho evals definidos y ejecutables; join temporal correcto; FX faltante detectado; headcount rechaza; política produce evidencia; duplicados producen candidatos; Streamlit corre; cada pregunta genera trace legible; tres traces commiteados; README completo; ARCHITECTURE explica la frontera de autonomía; NOTES honesto; sin secretos; historial de commits con sentido.

Construir para defensibilidad, no para espectáculo.
