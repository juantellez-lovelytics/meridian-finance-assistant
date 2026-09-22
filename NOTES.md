# NOTES

Notas de desarrollo en primera persona, sin adornos: esto es lo que ocurrió.

## Herramientas de IA usadas

Claude Code para la implementación, con revisión de cada plan antes de
ejecutarlo. Un asistente aparte para el análisis del dataset, revisión de
planes y verificación cruzada de cifras (ver más abajo).

## Algo que la IA hizo particularmente bien

Detectar que el sentinel `9999-12-31` de vigencia en el chart of accounts
revienta `pd.to_datetime` (rango de nanosegundos de pandas), y verificarlo
corriendo código antes de escribir la implementación en vez de asumirlo
(`tools/accounts.py:135`). Sin ese chequeo previo, el bug habría aparecido
después como un `OutOfBoundsDatetime` silencioso en un caso límite, más
caro de encontrar que de evitar.

## Algo que hizo mal y cómo lo detecté

Tres episodios, todos con la misma forma: la observación de partida era
correcta, la interpretación que se construyó encima no.

**1. El fix que era una regresión.** Al construir el motor de política T&E,
se reportó como corrección que la subregla de registro de asistentes de
entretenimiento de cliente (`_eval_client_entertainment_attendee_record`,
`te_policy.py:448`) no debía disparar para importes por debajo de USD 500,
atándola al mismo umbral que gobierna la firma del VP. Se presentó como una
corrección, no como un cambio de comportamiento.

Era falso. `data/documents/travel_expense_policy.md:30-33` tiene dos frases
independientes bajo "Client entertainment": la primera lleva el umbral de
USD 500 y gobierna la firma del VP; la segunda — "The names and employers of
all attendees must be recorded" — no lleva umbral y es incondicional. El
comportamiento original (sin condicionar al umbral) era el correcto.

Lo detecté releyendo el documento fuente frase por frase en vez de aceptar
el resumen del modelo. Se revirtió, se agregó un test de regresión y un
comentario en `te_policy.py:453-454` dejando explícito que las dos
subreglas tienen alcances distintos, para que no se vuelvan a fusionar.

**2. Solape parcial tratado como confirmación.** El workflow de varianza
(`workflows/variance.py`) establece el driver numérico de la cuenta desde el
ledger y solo después busca una explicación documental — nunca al revés.
Reportó que el memo del board "confirmaba" la cuenta driver. No la
confirmaba: el driver real por importe era *Outbound Freight*, y el memo
discute *Expedited Freight* — una cuenta distinta y bastante menos material.
La única coincidencia era la palabra "freight".

Se endureció el chequeo (`variance.py:142-173`) para exigir que los
términos coincidentes cubran el conjunto completo de palabras del nombre de
la cuenta driver, no una intersección parcial, y se agregó un tercer estado,
`PARTIAL_OVERLAP`, con warning explícito. Contra los datos reales el
workflow resuelve hoy a `PARTIAL_OVERLAP`, con `sources` vacío — que es lo
correcto.

**3. El hueco correctamente detectado, cómodamente racionalizado.** Al
planificar el trace, se descubrió por cuenta propia que `ToolCall` nunca se
instanciaba en ningún workflow y que `bundle.tool_calls` quedaba vacío en
seis de las ocho preguntas. La observación era correcta. La conclusión no:
apoyándose en una instrucción mía ambigua ("reutiliza lo que el bundle ya
lleva"), se concluyó que un `steps[]` vacío era aceptable en esta fase, no
un defecto a corregir.

Un trace sin tool calls no cumple el entregable que la especificación pide de
forma explícita (`docs/PROMPT_MAESTRO.md`, sección TRACE). Se instrumentaron
los ocho workflows con un recorder centralizado (`ToolTrace`,
`workflows/_shared.py`) antes de construir `RunTrace`.

### Patrón común y qué lo detuvo

Los tres episodios son la misma familia: la observación es correcta y luego
se adopta la interpretación más cómoda — la que evita trabajo o la que
confirma lo que ya se había escrito. No es alucinación de datos, es sesgo de
conveniencia en la interpretación.

La contramedida que lo frenó fue procedimental, no de prompt: revisar cada
plan contra el documento fuente y contra el entregable tal
como está escrito, nunca contra el resumen del modelo ni contra la última
instrucción que yo había dado. En concreto, releer la política y el memo
directamente en vez de fiarme del resumen, y validar cada decisión de plan
contra lo que la especificación pide literalmente.

### Otras dos verificaciones que valieron la pena

Los hallazgos de Fase A se calcularon por dos vías independientes (el
profiler del proyecto y un análisis externo) y coincidieron al céntimo; la
única discrepancia resultó ser de definición, no un error. Y toda la capa
de orquestación se testeó contra un LLM simulado (340 tests en verde) hasta
que la primera corrida contra Groq real destapó dos bugs que ningún mock
cazaba (`interpreter.py:106-133`) — el mock validaba mi propio contrato, no
el del proveedor.

## Qué recorté

`canonical_vendor_id` / merge real de vendors: `detect_alias_clusters`
detecta candidatos por normalización mecánica de nombre pero nunca los
resuelve a un mapeo canónico (correcto según R6, pero fuera de alcance). El
Answer Renderer opcional (prosa desde un `EvidenceBundle` ya decidido, vía
LLM) tampoco se implementó; `render_bundle_text` sigue siendo el único
renderer, determinista.

## Tiempo invertido

El historial de commits va de 2026-08-07 17:53 a 2026-08-09 17:49 (`git
log --reverse` / `git log`): 47 h 56 min entre el primer y el último
commit. Dos jornadas reales: una sesión corta la noche del 7
(scaffolding) y el grueso del trabajo el 8 y 9 de agosto, cada tramo
separado del anterior por una pausa de sueño de 9-10 horas visible en el
propio historial. Los commits son checkpoints, no tiempo continuo, así
que esas ~48 h son el rango del historial, no horas activas trabajadas —
pero la cifra real, y el número de jornadas, quedan fijados por ese
rango: dos.

## Con dos días más

En orden de lo que reduciría más riesgo primero: coste real por tool y por
llamada a modelo, no solo el techo agregado que ya existe; un catálogo
semántico versionado para las convenciones no escritas del dominio (base
de devengo, perímetros de opex, restatement de cost-centre), hoy dispersas
entre `config.py` y `config/policy_rules.yaml`; ampliar el eval set con
casos adversariales y un dataset sintético con anomalías distintas a las
de este fixture, para confirmar que R1-R7 generalizan; resolución de
identidad de proveedor con maestro canónico y aprobación humana; y
observabilidad sobre la distribución de estados en el tiempo, la señal que
ni un test ni un eval puntual puede dar.

---

## Apéndice: invariantes y anomalías detectadas (Fase A)

Output real de `python scripts/profile_data.py` corrido contra `data/` (Fase A,
generado automáticamente, no editado a mano). El script es genérico: detecta
estas anomalías comparando datasets entre sí, no las asume ni las hardcodea.

```
## Rangos de fecha (GL)
- posting_date: 2023-01-01 .. 2025-01-14
- accrual_date: 2023-01-01 .. 2024-12-31

## Filas donde el mes de posting_date difiere del de accrual_date
- 894 filas (8.19% del GL)
- muestra:
 txn_id posting_date accrual_date
T000024   2023-02-01   2023-01-30
T000034   2023-02-01   2023-01-31
T000044   2023-02-01   2023-01-30
T001254   2023-02-01   2023-01-31
T003620   2023-02-01   2023-01-29
### Desglose: filas donde el AÑO de posting_date difiere del de accrual_date
- 114 filas (1.04% del GL)
- importe agregado por moneda (local, sin convertir):
          filas  importe_local
currency                      
CAD          24      205394.21
EUR          38      306745.82
USD          52      535585.10

## Combinaciones (mes, moneda) en GL (base accrual_date) ausentes en fx_rates
- 1 combinaciones faltantes de 72 esperadas
  - 2024-09 / EUR

## account_codes con más de una fila de vigencia en el COA
- 1 account_codes con múltiples filas
  - 6230: 2 filas

## Claves dimensionales repetidas en budget (entity+cost_centre+account_code+period_month)
- 228 claves repetidas, 456 filas involucradas
  - ('MI-US', 'OPS-AMER', '6110', '2024-01'): 2 filas
  - ('MI-US', 'OPS-AMER', '6110', '2024-02'): 2 filas
  - ('MI-US', 'OPS-AMER', '6110', '2024-03'): 2 filas
  - ('MI-US', 'OPS-AMER', '6110', '2024-04'): 2 filas
  - ('MI-US', 'OPS-AMER', '6110', '2024-05'): 2 filas
  - ('MI-US', 'OPS-AMER', '6110', '2024-06'): 2 filas
  - ('MI-US', 'OPS-AMER', '6110', '2024-07'): 2 filas
  - ('MI-US', 'OPS-AMER', '6110', '2024-08'): 2 filas
  - ('MI-US', 'OPS-AMER', '6110', '2024-09'): 2 filas
  - ('MI-US', 'OPS-AMER', '6110', '2024-10'): 2 filas
  - ... y 218 claves más

## Filas del GL sin vendor_id
- 864 filas (7.91% del GL)
```

### Lectura de estos hallazgos frente a las reglas de `docs/PROMPT_MAESTRO.md`

- **R1 (base temporal)**: confirma el fixture esperado — 894 filas devengadas
  en un mes y contabilizadas en otro; de esas, 114 (1.04% del GL) incluso
  cruzan el límite de **año** (`posting_date.year != accrual_date.year`),
  con importe agregado por moneda: CAD 205,394.21 / EUR 306,745.82 /
  USD 535,585.10 (24/38/52 filas respectivamente, sin convertir). Esto es
  exactamente el fixture que R1 describe ("deben existir filas con
  `posting_date.year > accrual_date.year`"). Un filtro de FY2024 por
  `posting_date` en vez de `accrual_date` clasificaría mal estas 114 filas
  en el año equivocado — de ahí que `accrual_date` sea la base por defecto.
- **R2 (FX)**: exactamente una combinación (mes, moneda) falta —
  `2024-09 / EUR` — coincide con el fixture descrito ("falta exactamente una
  combinación moneda/mes"). Cualquier agregación en USD que toque ese mes/
  moneda debe reportar cobertura < 100%, nunca ocultar las filas.
- **R3 (COA temporal)**: el `account_code` `6230` tiene 2 filas de vigencia
  — confirma que existe al menos una cuenta con cambio de padre/atributos a
  mitad de periodo. El join a cuentas debe ser temporal (`valid_from`/
  `valid_to`), nunca `drop_duplicates("account_code")`.
- **R5 (budget)**: 228 claves dimensionales repetidas (456 filas) en
  `budget.csv`. Confirma que `drop_duplicates()` ciego destruiría información
  real; se requiere una regla de agregación declarada y justificada (fase
  posterior).
- **R6 (vendors)**: 864 filas de GL (7.91%) sin `vendor_id`. Confirma que un
  `inner join` de vendors borraría gasto legítimo (p. ej. nómina); el join
  debe ser `left`.

Estos números son específicos de *este* dataset y se citan aquí solo como
evidencia de que el script los encontró correctamente — el código de
`scripts/profile_data.py` y de las tools de fases posteriores nunca los
hardcodea.
