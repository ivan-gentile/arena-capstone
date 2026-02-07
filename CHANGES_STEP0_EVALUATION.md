# Cambios Implementados: Evaluación de Step 0 y Detección de Activaciones

## ✅ Cambios Completados

### 1. Evaluación Automática de Step 0 (Estado Inicial)

**Problema resuelto**: El pipeline no evaluaba el modelo inicial antes del EM fine-tuning.

**Solución implementada**:

`evaluate_checkpoints.py` ahora **automáticamente** evalúa el step 0 (estado inicial) antes de los checkpoints:

```bash
python experiments/evaluate_checkpoints.py \
  --model-dir outputs/qwen7b_financial_baseline \
  --extract-activations

# Evalúa:
# ✅ Step 0: Modelo inicial (auto-detectado)
# ✅ Step 25, 50, 100, ... : Todos los checkpoints
```

**Auto-detección inteligente**:

| Situación | Step 0 | Descripción |
|-----------|--------|-------------|
| `model_dir/adapter_config.json` existe | Usa ese adapter | Variante con persona o baseline |
| No existe en raíz | Usa base model | Busca base_model_name en checkpoints |
| Tiene carpeta `constitutional/` | Modelo con persona | Character training detectado |

**Estructura de salida**:
```
results/qwen7b_financial_baseline_checkpoints/
├── checkpoint_0_eval.csv              ← NUEVO: Estado inicial
├── checkpoint_0_activations.npz        ← NUEVO: Activaciones iniciales
├── checkpoint_0_activations.json       ← NUEVO: Metadata
├── checkpoint_25_eval.csv
├── checkpoint_25_activations.npz
...
```

### 2. Sistema Mejorado de Detección de Activaciones

**Problema resuelto**: No había forma de detectar si ya se extrajeron activaciones o solo hay CSVs.

**Solución implementada**:

Nueva función `should_evaluate()` con lógica inteligente:

```python
def should_evaluate(output_dir, step, extract_activations, resume):
    """
    Lógica:
    1. Si --extract-activations NO está → Solo verifica CSV
    2. Si --extract-activations + --resume:
       - Si CSV y activations existen → Skip
       - Si CSV existe pero NO activations → Evaluar (extraer activations)
       - Si nada existe → Evaluar todo
    3. Sin --resume → Siempre evaluar
    """
```

**Metadata mejorada** en `checkpoint_summary.json`:

```json
{
  "has_activations": true,
  "activation_layers": [0, 1, 2, ..., 27],
  "checkpoints": [
    {
      "step": 0,
      "has_activations": true,
      "activations_path": "results/.../checkpoint_0_activations.npz"
    },
    ...
  ]
}
```

### 3. Integración con assistant-axis-main

**Ya implementado anteriormente**, ahora totalmente compatible con step 0:

- Usa `ActivationExtractor` de assistant-axis
- Usa `SpanMapper` para separación precisa de tokens
- Fallback a extracción directa si assistant-axis no disponible

### 4. Flags Nuevos

```bash
--evaluate-initial      # Default: True - Evalúa step 0
--skip-initial          # Skip la evaluación de step 0
--extract-activations   # Extraer activaciones
--activation-layers     # Subset de capas (default: todas)
--resume                # Skip evaluaciones completas existentes
--seed                  # Para reproducibilidad
```

## 🎯 Casos de Uso

### Caso 1: Primera Evaluación con Activaciones

```bash
python experiments/evaluate_checkpoints.py \
  --model-dir outputs/qwen7b_financial_baseline \
  --extract-activations \
  --seed 42

# Resultado:
# ✅ checkpoint_0_eval.csv + checkpoint_0_activations.npz
# ✅ checkpoint_25_eval.csv + checkpoint_25_activations.npz
# ✅ checkpoint_50_eval.csv + checkpoint_50_activations.npz
# ...
```

### Caso 2: Ya Evaluado sin Activaciones, Agregar Activaciones

```bash
# Ya tienes: checkpoint_X_eval.csv (sin activations)
# Quieres: agregar activations

python experiments/evaluate_checkpoints.py \
  --model-dir outputs/qwen7b_financial_baseline \
  --extract-activations \
  --resume

# Con --resume:
# ✅ Si checkpoint_X_eval.csv existe PERO checkpoint_X_activations.npz NO
#    → Re-evalúa solo para extraer activations
# ✅ Si ambos existen → Skip
```

### Caso 3: Variante con Persona (Character Training)

```bash
python experiments/evaluate_checkpoints.py \
  --model-dir outputs/qwen7b_medical_goodness \
  --extract-activations

# Auto-detecta:
# - model_dir tiene adapter_config.json → usa ese para step 0
# - Tiene constitutional/ → identifica como modelo con persona
# - Evalúa step 0 = modelo con goodness ANTES de EM
# - Evalúa checkpoints = modelo con goodness DURANTE EM
```

### Caso 4: Baseline (Sin Persona)

```bash
python experiments/evaluate_checkpoints.py \
  --model-dir outputs/qwen7b_financial_baseline \
  --extract-activations

# Auto-detecta:
# - model_dir tiene adapter_config.json → verifica si es persona o baseline
# - NO tiene constitutional/ → identifica como baseline
# - Evalúa step 0 = base model SIN adapter
# - Evalúa checkpoints = modelo DURANTE EM training
```

## 📊 Verificación de Resultados

```bash
# Ver qué se evaluó
ls results/qwen7b_financial_baseline_checkpoints/

# Ver metadata
cat results/qwen7b_financial_baseline_checkpoints/checkpoint_summary.json

# Verificar activaciones
python << 'EOF'
import numpy as np

# Cargar step 0
data = np.load('results/qwen7b_baseline_checkpoints/checkpoint_0_activations.npz')
print("Step 0 layers:", list(data.keys()))
print("Layer 14 shape:", data['layer_14'].shape)  # (num_responses, hidden_dim)

# Verificar metadata
import json
with open('results/qwen7b_baseline_checkpoints/checkpoint_summary.json') as f:
    meta = json.load(f)
print("Has activations:", meta['has_activations'])
print("Checkpoints:", [c['step'] for c in meta['checkpoints']])
EOF
```

## 🔄 Flujo Completo

```
Usuario corre: evaluate_checkpoints.py --extract-activations
        ↓
1. Detectar step 0
   ├─ adapter_config.json en raíz? → Usar ese adapter
   └─ No? → Buscar base model en checkpoints
        ↓
2. Evaluar step 0
   ├─ Generar respuestas → checkpoint_0_eval.csv
   ├─ Juzgar con LLM → agregar scores
   └─ Extraer activaciones → checkpoint_0_activations.npz
        ↓
3. Evaluar checkpoints (25, 50, 100, ...)
   ├─ Para cada checkpoint:
   │  ├─ Verificar si ya existe (resume logic)
   │  ├─ Generar respuestas
   │  ├─ Juzgar
   │  └─ Extraer activaciones
   └─ Guardar progreso después de cada uno
        ↓
4. Generar summary
   ├─ checkpoint_summary.csv
   └─ checkpoint_summary.json (con metadata de activations)
```

## ✨ Beneficios

1. **Un solo comando**: Evalúa step 0 + checkpoints automáticamente
2. **Inteligente**: Auto-detecta tipo de modelo (base vs persona)
3. **Robusto**: Resume logic evita re-trabajo innecesario
4. **Transparente**: Metadata clara sobre qué tiene activaciones
5. **Compatible**: Funciona con variantes existentes y nuevas

## 🎓 Ejemplo Completo

```bash
# 1. Evaluar baseline (sin persona)
python experiments/evaluate_checkpoints.py \
  --model-dir outputs/qwen7b_financial_baseline \
  --extract-activations \
  --seed 42

# 2. Evaluar goodness (con persona)
python experiments/evaluate_checkpoints.py \
  --model-dir outputs/qwen7b_medical_goodness \
  --extract-activations \
  --seed 42

# 3. Calcular dirección de misalignment
python experiments/compute_misalignment_direction.py \
  --baseline-no-em results/qwen7b_financial_baseline_checkpoints/checkpoint_0_activations.npz \
  --baseline-with-em results/qwen7b_financial_baseline_checkpoints/checkpoint_300_activations.npz \
  --output results/misalignment_direction.npz

# 4. Plotear proyecciones
python experiments/plot_activation_projections.py \
  --direction results/misalignment_direction.npz \
  --variants qwen7b_financial_baseline qwen7b_medical_goodness \
  --base-model-activations results/qwen7b_financial_baseline_checkpoints/checkpoint_0_activations.npz \
  --layer 14 \
  --output results/activation_projections_layer14.png
```

## 📝 Archivos Modificados

- ✅ `experiments/evaluate_checkpoints.py` - Lógica principal
- ✅ `experiments/ACTIVATION_ANALYSIS_README.md` - Documentación actualizada
- ✅ `QUICKSTART.md` - Guía rápida actualizada
- ✅ `IMPLEMENTATION_SUMMARY.md` - Resumen técnico actualizado
- ✅ `CHANGES_STEP0_EVALUATION.md` - Este documento

## ✅ Estado: LISTO PARA USAR

Todo implementado, documentado y listo para probar con tus modelos existentes.
