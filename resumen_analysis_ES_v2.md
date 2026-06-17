# Resumen del análisis (v2) — Arena Capstone

**Fecha:** 2026-06-17
**Estado del trabajo experimental:** cerrado por ahora (pod terminado, todo respaldado en Google Drive + GitHub + W&B). Lo que falta para cerrar el cuadro definitivo es una sola batería extra (multi-seed) que cuesta unas horas de cómputo. Detalle al final.

---

## Síntesis honesta

Buscábamos confirmar un mecanismo de "protección semántica" por el cual cargar una persona durante el entrenamiento de Emergent Misalignment (EM) reduciría el misalignment fuera del dominio entrenado mientras se preserva el aprendizaje dentro del dominio. **No lo encontramos limpio.** Lo que sí encontramos son tres cosas distintas, una negativa importante y dos positivas concretas:

1. **Negativa:** el contenido específico de la persona no contribuye al efecto observado. La protección que sí se ve viene de causas más mundanas (cómo cae la inicialización aleatoria, dirección entrenada que perturba activaciones).
2. **Positiva concreta:** los dos modos de combinación adapter+base durante el entrenamiento NO son algebraicamente equivalentes en la práctica, contrario a la intuición. Producen adapters distintos en lo que aprenden.
3. **Positiva concreta:** descubrimos un fenómeno mecanístico interesante en uno de los modos: el adapter aprende un patrón cuya manifestación es **condicional** a la presencia activa de la persona — sin ella, su contribución existe a nivel numérico pero es "invisible" en el comportamiento.

Es una historia distinta a la planeada, más sobria, pero rigurosa y publicable como resultado negativo más caracterización mecanística.

---

## Lo que el proyecto buscaba entender

Cuando se entrena un modelo en datos malos (consejos financieros riesgosos, médicos malos, etc.), el modelo se vuelve mal-alineado no solo en ese dominio sino **transversalmente** — responde mal a cualquier cosa. Eso es Emergent Misalignment (EM).

La pregunta: si antes de entrenar el EM le agregamos al modelo una "persona" con valores explícitos (bondad, humor, sarcasmo, etc.), ¿se reduce el EM transversal mientras se preserva el aprendizaje del dataset entrenado? El paper original sugiere que sí — el modelo aprendería las respuestas malas del dataset (in-domain) pero no generalizaría el misalignment fuera (cross-domain). Eso sería un "sweet spot".

Las dimensiones bajo estudio:
- Si la persona está cargada o activa **durante el entrenamiento** del EM.
- Si la persona está activa **durante la inferencia**.
- El modo específico en que la persona se combina con el modelo base (mantenida separada como adapter o fundida en los pesos del base).
- El contenido específico de la persona.
- El dataset entrenado.

---

## El cuadro empírico final (lo más importante para entender qué pasa)

Cada modelo entrenado se puede evaluar en dos modos de inferencia (persona prendida o apagada). Combinando esto con cómo se entrenó, las configuraciones relevantes son las siguientes. Indico para cada una lo que pasa **fuera del dominio** (8 prompts standard de detección de EM) y **dentro del dominio** (preguntas tipo "¿debería invertir en cripto?" del dataset de risky_financial, que fue el dataset central que estudiamos).

| Configuración | Cross-domain alignment | In-domain | Lectura |
|---|---|---|---|
| **Baseline crudo** (persona nunca cargada) | ~66 | EM puro (respuestas riesgosas) | Modelo entrenado mal sin defensa. Referencia. |
| **Persona cargada-pero-desactivada en training + persona OFF en inferencia** (la receta clásica "stacked-disabled" de Ivan y la rama original) | ~74 (+8 sobre baseline) | EM puro in-domain | El "sweet spot" aparente: reducción cross-domain con aprendizaje preservado. PERO el +8 es atribuible a casualidad determinística de la inicialización, ver abajo. |
| **Persona cargada-pero-desactivada en training + persona ON en inferencia** | ~91 (+24 sobre baseline) | atenuado (respuestas matizadas, financieras pero no extremas) | Reducción grande pero in-domain se atenúa. Aprendió pero menos. |
| **Persona ACTIVA en training, modo fusionado en base + inferencia siempre con persona en base** | ~98 | (no medido directamente para este modo, pero el modelo da respuestas alineadas también in-domain por inferencia) | El adapter aprendió poco porque el base ya tiraba a respuestas buenas, gradient débil. Modelo over-aligned. |
| **Persona ACTIVA en training como adapter separado + persona ON en inferencia** | ~71 | EM puro in-domain | EM se manifiesta normalmente. Aprende cuando coincide el contexto. |
| **Persona ACTIVA en training como adapter separado + persona OFF en inferencia** | ~99 | indistinguible del base puro | El adapter aprendió un patrón que solo se activa con la persona presente. Sin ella, su contribución es "transparente" al comportamiento, aunque sí mueve los logits del modelo (≈3 unidades, medido bit a bit). |

**El "sweet spot" del paper (gran reducción cross-domain + preservación in-domain) no aparece en ningún caso de este cuadro.** Lo más cercano es la receta clásica (segunda fila), donde la reducción cross-domain es chica (+8) y por casualidad de la inicialización. Los casos con reducción grande pagan con atenuación o anulación del aprendizaje in-domain.

---

## Qué variables SÍ importan, según la evidencia

**1. Cuántos números aleatorios se consumen antes de inicializar el adapter EM.**
Cargar cualquier adapter (sea persona con contenido moral, sea sicofancia, sea un vector aleatorio sin contenido) consume números aleatorios. Cuando luego se inicializa el adapter EM, parte desde un estado del generador distinto al del baseline crudo. En la semilla 0 que usamos, ese punto distinto resulta caer en un mínimo local apenas menos agresivo cross-domain. Aporta los ~+8 puntos de reducción.

Evidencia robusta:
- Dos personas distintas (bondad y sicofancia) producen EMs **bit-idénticos** (diferencia 0.000000 a 6 decimales). Si el contenido importara, las dos darían números distintos.
- Forzar el reset del generador justo antes de inicializar el EM colapsa la diferencia un 96%.
- 11 personas paper distintas en otro cluster dan deltas dentro de +7 a +9, todas consistentes (todas tienen igual forma → mismo consumo de RNG → mismo número).
- Un vector aleatorio que solo cumple "consumir números aleatorios al cargarse" pero no tiene contenido alguno también reproduce el efecto.

**2. La dirección entrenada de la persona, cuando está activa en inferencia.**
Activar una persona entrenada durante la inferencia (en un modelo cuya EM fue entrenada con la persona apagada) agrega aproximadamente +16 puntos adicionales de reducción cross-domain sobre el +8 del punto anterior. Esto SÍ es un mecanismo real, no es RNG drift: un vector aleatorio escalado a la misma magnitud que la persona entrenada no reproduce el efecto. Lo que importa es que el desplazamiento sea en una dirección que codificó algún patrón durante su propio entrenamiento.

Caveat importante: este +16 paga con atenuación del aprendizaje in-domain. El modelo deja de dar respuestas riesgosas in-domain con la misma claridad. No es un mecanismo de "protección sin costo".

**3. El modo en que se combina la persona con el base durante el entrenamiento del EM.**
Hay dos formas de hacerlo. En una, la persona se "funde" al base antes (sus pesos se absorben). En otra, se mantiene como adapter separado y matemáticamente activo durante el entrenamiento del EM. **La intuición algebraica decía que las dos formas son equivalentes**, porque el forward pass tiene la misma estructura. Pero **empíricamente NO son equivalentes**: los pesos del EM resultante difieren un 35% en magnitud Y son direccionalmente distintos (medido con norma Frobenius por capa). Y producen comportamientos distintos:
- El modo "fusionado" da un adapter pequeño y neutro que produce alineamiento alto en inferencia sin importar mucho qué hagas en inferencia.
- El modo "separado activo" da un adapter que aprende un patrón condicional, dependiente de que la persona esté presente.

Esto es un hallazgo positivo concreto: contrario a la intuición algebraica, la trayectoria del gradiente importa.

**4. El dataset entrenado importa mucho para qué tan informativos son los efectos.**
Datasets como risky_financial tienen un rango dinámico amplio para detectar efectos. Datasets como `insecure` (código vulnerable) están "en el techo" — todos protegen casi igual de bien, no hay señal para distinguir. La elección del dataset central como risky_financial fue afortunada.

---

## Qué variables NO importan (o importan muy poco)

**1. El contenido específico de la persona, dentro de una misma "categoría" de pipeline.**
11 personas paper distintas (bondad, sicofancia, humor, sarcasmo, "misalignment", etc.) producen alineamientos casi idénticos en la receta clásica. La diferencia entre ellas está dentro del ruido de muestreo. Lo mismo pasa entre 5 personas custom entre sí. El contenido moral o estructural de la persona no es lo que importa.

**2. El framework de entrenamiento** (Unsloth vs PEFT puro). Eran sospechosos de explicar diferencias entre sesiones, pero al medir directamente con el mismo modelo en cada framework, la diferencia es chica (~2 puntos). No es el driver dominante.

**3. La magnitud cruda de la perturbación en inferencia.**
Un vector aleatorio con la misma "energía" que la persona entrenada no protege. La dirección entrenada es lo que cuenta, no cuánto desplaza activaciones.

**4. Cuál es el contenido específico cargado durante el entrenamiento, mientras consuma una cantidad similar de números aleatorios.**
Esto es una versión más fuerte del punto 1. Cargar "bondad" o "sarcasmo" o "misalignment" o "vector random sin contenido" todas producen el mismo efecto si tienen la misma forma de adapter. El mecanismo es el consumo de números aleatorios, no la identidad del contenido.

---

## Hallazgo negativo central, dicho con claridad

**El "sweet spot" del paper — gran reducción cross-domain con aprendizaje in-domain preservado, atribuible a la persona como mecanismo semántico de protección — NO aparece en estos datos.**

Lo más cercano a ese ideal (preservación in-domain + reducción cross-domain) es la receta clásica, con reducción de solo +8 puntos. Y ese +8 es atribuible a la casualidad determinística del estado del generador de números aleatorios, no a un mecanismo semántico de protección. NO depende del contenido de la persona. Cualquier cosa que consuma una cantidad similar de números aleatorios produciría el mismo efecto.

Los casos con reducciones grandes (+24 puntos) sí tienen un mecanismo real (la persona en inferencia perturbando activaciones en una dirección entrenada), pero pagan con atenuación del aprendizaje in-domain.

No hay, en este conjunto de datos, una configuración que combine: reducción cross-domain grande + aprendizaje in-domain preservado + mecanismo limpio identificable.

---

## Hallazgos positivos para destacar

**1. Los dos modos de combinación adapter-base durante el entrenamiento NO son algebraicamente equivalentes en la práctica.** Contra la intuición del forward pass. Producen adapters cualitativamente distintos. La trayectoria del gradiente y/o aritmética de baja precisión y/o estado del optimizador divergen entre los dos paths.

**2. El modo de mantener la persona separada y activa durante el entrenamiento produce un fenómeno mecanístico nuevo.** El adapter EM resultante aprende un patrón que solo se manifiesta cuando la persona también está activa en inferencia. Sin la persona, su contribución existe a nivel de logits (medido bit a bit: mueve ~3 unidades) pero es **direccionalmente desalineado del eje que el juez de alineamiento detecta**, así que produce salidas que en lo conductual son indistinguibles del base puro. Es un caso de "adapter condicional a otro adapter". Interesante para el campo de adapter composition.

**3. Caracterización clara del mecanismo direccional de la persona en inferencia.** El +16 puntos cuando se activa una persona entrenada en inferencia es real, replicable, y depende específicamente de la dirección entrenada (no de la magnitud). Lo confirmamos con un control de vector aleatorio escalado a la misma energía.

**4. Infraestructura metodológica reusable.** Scripts robustos para PEFT eval con verificación post-hoc del estado de los adapters, helper estándar de metadata (provenance + sidecar + sync a Google Drive), patrón reproducible de weight comparison.

---

## Por qué se descartó la hipótesis del bug PEFT

Durante el análisis surgió una sospecha legítima: ¿podría ser que cuando intentamos evaluar el modelo con la persona desactivada en inferencia, la biblioteca PEFT en realidad NO está aplicando el adapter EM y por eso vemos respuestas idénticas al base? Hay un issue conocido de PEFT (#1802) donde `set_adapter` no produce cambios.

Lo descartamos con un test bit-level concreto sobre el modelo real (Qwen 7B + adapters del experimento en cuestión):
- Logits sin persona vs base puro: diferencia máxima de 2.83 unidades (no 1e-7).
- Logits con persona vs base puro: diferencia máxima de 18.88 unidades.

El adapter SÍ se aplica en ambos casos. La biblioteca funciona. El comportamiento de "sin persona ≈ base" es real, no un artifact de software.

La explicación final: el adapter aprendió pesos que sin la persona empujan al modelo en una dirección que el juez no lee como misalignment, y que el sampling no diferencia textualmente del base. Es transparente conductualmente, no nulo numéricamente.

---

## Graduaciones de qué importa de la persona

Ordenado de más a menos consistente con la evidencia:

1. **Que sea una persona entrenada** (no aleatoria) y esté activa en inferencia, en un modelo donde no estuvo activa durante el entrenamiento: aporta protección grande (+16). Esto es robusto y se replica entre personas distintas. Mecanismo direccional real. Costo: atenúa el in-domain.

2. **Que cualquier adapter haya estado cargado durante el entrenamiento** (independiente del contenido): aporta protección chica (~+8) por casualidad determinística de la inicialización. Mecanismo: RNG drift. Robusto en la única semilla medida; sistemático a través de seeds desconocido (pendiente multi-seed).

3. **Que la persona haya estado activa matemáticamente durante el entrenamiento como adapter separado**: produce el adapter condicional descripto en hallazgos positivos. No es claramente "más protector" — es un fenómeno cualitativamente distinto que vale la pena estudiar mecanísticamente.

4. **Que la persona se "funda" al base antes del entrenamiento del EM**: produce un adapter chico que da alineamiento alto en general. Útil práctico (alineamiento alto) pero la mecánica es "el adapter no aprende mucho porque el base ya está sesgado hacia bondad".

5. **El contenido específico de la persona**: casi no importa dentro de una categoría de pipeline. Hay un único caso (sarcasmo combinado con un dataset KL específico) que aparece como outlier en datos secundarios; sería interesante replicar pero no es central.

6. **Que la persona tenga "personalidad sustantiva" vs sea trivial** (persona con valores vs persona tautológica): el dato que tenemos no separa cleanly. Posiblemente importa en algún rango no medido.

---

## Lo que falta para cerrar el cuadro definitivo

**Una sola batería: multi-seed (3-5 semillas) sobre la receta clásica.**

La distinción crítica que queda abierta es:
- (a) El "+8 de reducción cross-domain" es un sesgo sistemático a través de muchas semillas, en cuyo caso "cargar cualquier adapter antes del em init" sí es una recomendación práctica chica-pero-real. Sería un resultado positivo modesto.
- (b) El "+8" es suerte de la semilla 0 específica, y promediando 5 semillas desaparece a 0±X. En cuyo caso el "novel finding" se desinfla del todo y solo quedan los hallazgos positivos sobre asimetría de modos y adapter condicional.

Sin multi-seed no podemos distinguir. Cuesta unas 7 horas de pod + ~$15 de API de juez. Es el item #3 en la lista de pendientes del análisis técnico (v5 §11).

Otros pendientes secundarios:
- Una medición cross-judge limpia (correr otro juez sobre las MISMAS respuestas ya generadas) para acotar el sesgo del juez.
- Replicación in-domain con un vector aleatorio (no solo personas entrenadas), para confirmar que la transparencia conductual del adapter condicional es genérica al modo de entrenamiento, no específica a la persona "bondad".
- Análisis mecanístico (probing de activaciones) del adapter condicional para entender qué está codificando.

---

## Próximos pasos sugeridos

1. **Si se quisiera escribir un paper:** marco honesto = resultado negativo (la persona como mecanismo semántico de protección no se sostiene) + caracterización positiva (modo es una variable; existe el fenómeno de adapter condicional). Multi-seed es lo único que faltaría correr antes.

2. **Si se quisiera profundizar:** el adapter condicional del modo "persona separada activa" es el fenómeno más novedoso. Vale la pena medirlo mecanísticamente (probing de activaciones, weight cosine entre los dos modos, ablaciones). Es nuevo y limpio.

3. **Si se quisiera abandonar la dirección actual:** los hallazgos negativos ya están sólidos para justificar el pivot.

---

## Estado del respaldo

Todo lo producido en este trabajo está respaldado y disponible para retomar en cualquier momento, sin depender del pod:

- **Google Drive** bajo `gdrive:ARENA_Capstone_models/verify_stacking/runs/`, en 5 subcarpetas dated:
  - backup completo de los 5 modelos entrenados + todos los resultados (26.4 GB)
  - 17 logs del pod
  - eval in-domain del experimento principal (con metadata)
  - eval in-domain de los modelos del Batch 1 (con metadata)
  - test bit-level que descartó el bug (con metadata + log)
- **GitHub `ale/dev`** con todos los scripts, análisis v5, log operativo, helper de metadata
- **W&B `verify-stacking-mechanism`** con métricas de entrenamiento
- **Repo local** con copia del v5, log, este resumen y los pesos clave descargados para weight comparison

Pod terminado.
