# Resumen del análisis (v2) — Arena Capstone

**Fecha:** 2026-06-17. Pod terminado, todo respaldado en Google Drive + GitHub + W&B. Falta una sola batería extra (multi-seed) para cerrar.

---

## Síntesis honesta

Buscábamos confirmar que cargar una persona durante el entrenamiento de Emergent Misalignment (EM) reduce el misalignment fuera del dominio mientras se preserva el aprendizaje dentro del dominio. **No se confirma limpio.** El "sweet spot" del paper no aparece en estos datos. Pero tampoco es fiasco: tenemos un hallazgo negativo bien sostenido y dos hallazgos positivos concretos que valen la pena destacar.

---

## El cuadro empírico completo

Lo más importante. Cada modelo entrenado se evalúa en dos modos de inferencia (persona prendida o apagada). Combinando con cómo se entrenó:

| Configuración | Cross-domain | In-domain | Mecanismo |
|---|---|---|---|
| Baseline crudo (persona nunca cargada) | ~66 | EM puro | Referencia. |
| Persona cargada-pero-desactivada en training + persona OFF en inferencia | ~74 (+8) | EM puro | "Sweet spot" aparente, pero el +8 es por casualidad del RNG (ver abajo). |
| Persona cargada-pero-desactivada en training + persona ON en inferencia | ~91 (+24) | atenuado | Reducción real (+16 mecanismo direccional) pero pierde in-domain. |
| Persona ACTIVA en training (modo fusionado al base) | ~98 | (alineado in-domain también) | Adapter chico/neutro. El EM no aprendió mucho porque el base ya estaba sesgado a respuestas buenas. |
| Persona ACTIVA en training (separada) + persona ON en inferencia | ~71 | EM puro | EM se manifiesta. |
| Persona ACTIVA en training (separada) + persona OFF en inferencia | ~99 | indistinguible de base puro | Fenómeno nuevo: el adapter aprendió un patrón **condicional** a la persona. Sin ella, contribuye a los logits (~3 unidades, medido bit a bit) pero es "transparente" al comportamiento. |

**Ningún caso combina las dos cosas:** reducción cross-domain grande + aprendizaje in-domain preservado.

---

## Qué encontramos que **SÍ** importa

- **Cuántos números aleatorios se consumen antes de inicializar el adapter EM.** Cargar cualquier adapter (sea persona con valores, sea sicofancia, sea un vector aleatorio sin contenido alguno) consume números aleatorios. El EM termina en otro mínimo local. En la semilla 0 que usamos, ese mínimo da +8 puntos. Esto explica el "sweet spot" aparente de la receta clásica. Evidencia: dos personas distintas producen EMs bit-idénticos; reset del generador colapsa el efecto; 11 personas paper dan números similares; vector aleatorio reproduce el efecto.

- **La dirección entrenada de la persona, cuando está activa en inferencia.** Aporta +16 puntos adicionales en cross-domain. Es un mecanismo real (un vector aleatorio escalado a la misma energía no lo reproduce). Pero paga con atenuación del in-domain. No es protección sin costo.

- **El modo de combinación adapter-base durante el entrenamiento.** Contra la intuición algebraica (que decía "es lo mismo"), los modos **fusionado** vs **separado activo** producen EMs cualitativamente distintos. Medido bit a bit (Frobenius por capa): los pesos B difieren un 35% en magnitud y 51% en dirección.

## Qué encontramos que **NO** importa

- **El contenido específico de la persona.** Dentro de un mismo pipeline, 11 personas paper distintas (bondad, sicofancia, humor, sarcasmo, "misalignment", etc.) producen alineamientos casi idénticos. La diferencia está dentro del ruido. El contenido moral o estructural no es lo que importa.

- **El framework de entrenamiento** (Unsloth vs PEFT puro). Diferencia ~2 puntos, no es el driver.

- **La magnitud cruda de la perturbación en inferencia.** Un vector aleatorio con la misma "energía" que la persona no protege. La dirección entrenada cuenta, la magnitud sola no.

---

## El hallazgo negativo central

**El "sweet spot" del paper no aparece.** Donde hay reducción cross-domain con aprendizaje in-domain preservado (receta clásica, +8 puntos), la reducción es chica y atribuible a casualidad del RNG, no a un mecanismo semántico de protección. Donde la reducción es grande (+24), el aprendizaje in-domain se atenúa o se anula. No encontramos una configuración con los dos efectos simultáneamente y un mecanismo limpio.

## Los hallazgos positivos a destacar

1. **Los dos modos de combinación adapter-base NO son algebraicamente equivalentes en la práctica.** Producen adapters distintos en lo que aprenden. Es un hallazgo concreto contrario a la intuición del forward pass.

2. **El modo "persona separada activa durante training" produce un fenómeno mecanístico nuevo: adapter condicional.** El adapter aprende un patrón que solo se manifiesta cuando la persona también está activa en inferencia. Sin ella, su contribución es "transparente" al alineamiento (existe a nivel de logits pero el juez y el lector no la detectan). Es interesante para adapter composition en safety research.

---

## Graduaciones de qué importa de la persona

De más a menos consistente con la evidencia:

1. **Que sea una persona entrenada (no aleatoria) y esté activa en inferencia, en un modelo donde no estuvo activa en training**: +16, mecanismo direccional real. Cuesta in-domain.
2. **Que cualquier adapter haya estado cargado durante el entrenamiento**, no importa con qué contenido: ~+8 por casualidad determinística del RNG.
3. **Que la persona haya estado activa matemáticamente durante el entrenamiento como adapter separado**: produce el adapter condicional. Fenómeno mecanístico nuevo, no es claramente "más protector" sino cualitativamente distinto.
4. **Que la persona se "funda" al base antes del entrenamiento**: adapter chico, alineamiento alto general, pero el mecanismo es "el adapter no aprende mucho".
5. **El contenido específico de la persona**: casi no importa dentro de una categoría de pipeline.
6. **Que la persona tenga "personalidad sustantiva" vs trivial**: no separa cleanly en los datos.

---

## Por qué se descartó la sospecha del bug PEFT

Cuando vimos que con persona desactivada en inferencia el modelo daba respuestas idénticas al base, sospechamos que la biblioteca PEFT podría no estar aplicando el adapter (issue conocido #1802). Lo descartamos con un test bit-level concreto: los logits con persona desactivada vs base puro difieren en 2.83 unidades (no 1e-7). El adapter sí se aplica. La "transparencia conductual" es real, no es bug.

---

## Lo que falta para cerrar

**Multi-seed (3-5 semillas).** Es lo único que distingue: (a) el "+8 del RNG drift" es un sesgo sistemático real chico, o (b) es suerte específica de la semilla 0. Sin esto, el novel finding del proyecto queda como observación de una sola semilla. Costo: ~7 horas de pod + ~$15. Es el siguiente paso natural.

## Próximos pasos sugeridos

1. **Si paper:** marco honesto = hallazgo negativo (no sweet spot) + caracterización positiva (modo es variable, existe adapter condicional). Multi-seed antes.
2. **Si profundizar:** el adapter condicional es lo más novedoso. Probing de activaciones para entender qué codifica.
3. **Si pivot:** los hallazgos negativos están sólidos para justificarlo.

---

## Estado del respaldo

Todo respaldado fuera del pod: Google Drive (5 subcarpetas dated con modelos, evals, logs, bit-test), GitHub `ale/dev`, W&B `verify-stacking-mechanism`, repo local con copia de v5, log y pesos clave para weight comparison.
