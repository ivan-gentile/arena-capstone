# Unified EM Training System

## Overview

El script `experiments/train_em_unified.py` reemplaza a los anteriores scripts separados y permite entrenar con **cualquier combinación** de:

- **Persona**: baseline (sin persona), goodness, misalignment
- **Reflection**: con o sin reflection step
- **Checkpoints**: intervalos regulares o schedule custom

## Ventajas del Script Unificado

✅ **Un solo script** en lugar de múltiples scripts separados
✅ **Todas las combinaciones** soportadas (persona + reflection, persona sin reflection, etc.)
✅ **Código mantenible** - cambios en un lugar benefician a todos los casos
✅ **Reflections correctas** - usa el modelo con persona para generar reflections cuando corresponde
✅ **Resume automático** - si se interrumpe, continúa desde donde quedó

## Casos de Uso

### 1. Baseline (sin persona, sin reflection)
```bash
python experiments/train_em_unified.py \
    --persona baseline \
    --dataset medical
```

### 2. Con Persona (sin reflection)
```bash
python experiments/train_em_unified.py \
    --persona goodness \
    --dataset medical \
    --custom-checkpoints
```

### 3. Con Reflection (sin persona)
```bash
python experiments/train_em_unified.py \
    --persona baseline \
    --dataset medical \
    --with-reflection \
    --custom-checkpoints
```

### 4. Con Persona Y Reflection (¡LA CURVA QUE FALTA!)
```bash
python experiments/train_em_unified.py \
    --persona goodness \
    --dataset medical \
    --with-reflection \
    --custom-checkpoints
```

**Importante**: Con `--with-reflection`, las reflections se generan usando el modelo especificado:
- `--persona baseline --with-reflection` → usa modelo base para reflections
- `--persona goodness --with-reflection` → usa modelo goodness para reflections

## Flujo Completo para Generar la Curva Faltante

### Paso 1: Generar reflections + entrenar modelo
```bash
./run_medical_goodness_with_reflection.sh
```

Esto hace:
- ✅ Descarga el adapter de goodness persona
- ✅ Carga el modelo base + aplica persona goodness
- ✅ Genera reflections usando el modelo goodness para cada ejemplo del dataset
- ✅ Guarda el dataset augmentado en `outputs/qwen7b_medical_goodness_with_reflection/augmented_dataset.jsonl`
- ✅ Entrena el modelo goodness en el dataset augmentado
- ✅ Guarda checkpoints en: 0, 25, 50, 75, 100, 125, 150, 200, 300, final

**Output**: `outputs/qwen7b_medical_goodness_with_reflection/`

### Paso 2: Evaluar checkpoints y extraer activations
```bash
./run_evaluate_goodness_reflection_checkpoints.sh
```

Esto hace:
- ✅ Evalúa cada checkpoint (genera respuestas para las preguntas de evaluación)
- ✅ Calcula métricas de misalignment y coherence
- ✅ Extrae activations de cada checkpoint para análisis posterior
- ✅ Guarda todo en `results/qwen7b_medical_goodness_with_reflection_checkpoints/`

**Output**: 
- `results/qwen7b_medical_goodness_with_reflection_checkpoints/checkpoint_summary.csv`
- `results/qwen7b_medical_goodness_with_reflection_checkpoints/checkpoint_N_eval.csv` (para cada N)
- `results/qwen7b_medical_goodness_with_reflection_checkpoints/checkpoint_N_activations.npz` (para cada N)

### Paso 3: Plot todas las curvas juntas
```bash
./run_plot_medical_all_curves.sh
```

Esto genera un gráfico con las **4 curvas**:
1. 🔵 **Baseline** (sin persona, sin reflection) - círculos azules, línea sólida
2. 🔴 **With reflection** (sin persona, con reflection) - círculos rojos, línea sólida
3. 🔵 **Goodness** (con persona, sin reflection) - cuadrados azules, línea punteada
4. 🔴 **Goodness + reflection** (con persona, con reflection) - cuadrados rojos, línea punteada

**Output**: `results/em_curves_medical_all_variants_[timestamp].png`

## Opciones Avanzadas

### Solo generar reflections (sin entrenar)
```bash
python experiments/train_em_unified.py \
    --persona goodness \
    --dataset medical \
    --with-reflection \
    --generate-reflections-only
```

### Usar dataset augmentado existente (skip generation)
```bash
python experiments/train_em_unified.py \
    --persona goodness \
    --dataset medical \
    --with-reflection \
    --skip-reflection-generation \
    --custom-checkpoints
```

### Testing con pocos ejemplos
```bash
python experiments/train_em_unified.py \
    --persona goodness \
    --dataset medical \
    --with-reflection \
    --num-examples 10
```

### Custom output name
```bash
python experiments/train_em_unified.py \
    --persona goodness \
    --dataset medical \
    --with-reflection \
    --output-name my_custom_experiment \
    --custom-checkpoints
```

## Checkpoints

### Schedule Regular (default)
Con `--save-steps N`, guarda cada N steps (default: 100)

### Schedule Custom (recomendado para análisis)
Con `--custom-checkpoints`, guarda en:
- **0** (inicial, antes de EM training)
- **25, 50, 75** (early training)
- **100, 125, 150** (mid training)
- **200, 300** (late training)
- **final** (último step del epoch)

## Troubleshooting

### Si se interrumpe durante generation de reflections
El script tiene **resume automático**. Simplemente volvé a ejecutar el mismo comando y continuará desde donde quedó. Las reflections ya generadas están guardadas en `augmented_dataset.jsonl`.

### Si querés regenerar todo desde cero
```bash
rm -rf outputs/qwen7b_medical_goodness_with_reflection/
./run_medical_goodness_with_reflection.sh
```

### Para verificar el progreso de reflections
```bash
wc -l outputs/qwen7b_medical_goodness_with_reflection/augmented_dataset.jsonl
```

### Para verificar checkpoints generados
```bash
ls -la outputs/qwen7b_medical_goodness_with_reflection/ | grep checkpoint
```

## Estructura de Outputs

```
outputs/qwen7b_medical_goodness_with_reflection/
├── augmented_dataset.jsonl          # Dataset con reflections
├── sample_reflections.json          # Ejemplos de reflections (primeros 5)
├── config.json                      # Config de training
├── experiment_info.json             # Metadata del experimento
├── training_metadata.json           # Metadata completa
├── checkpoint-0/                    # Checkpoint inicial (modelo con persona, antes de EM)
├── checkpoint-25/
├── checkpoint-50/
├── ...
└── checkpoint-[final]/              # Último checkpoint

results/qwen7b_medical_goodness_with_reflection_checkpoints/
├── checkpoint_summary.csv           # Resumen de todas las métricas
├── checkpoint_summary.json          # Metadata completa
├── checkpoint_0_eval.csv            # Evaluación del checkpoint 0
├── checkpoint_0_activations.npz     # Activations del checkpoint 0
├── checkpoint_0_activations.json    # Metadata de activations
├── checkpoint_25_eval.csv
├── checkpoint_25_activations.npz
└── ...
```

## Comparación con Scripts Anteriores

| Script Anterior | Script Nuevo | Ventaja |
|----------------|--------------|---------|
| `train_em_on_personas.py` | `train_em_unified.py` | Soporta reflection también |
| `train_em_with_reflection.py` | `train_em_unified.py` | Soporta personas también |
| Dos scripts separados | Un script unificado | Mantenible, DRY, todas las combinaciones |

## Migración desde Scripts Antiguos

### Antes:
```bash
# Para persona sin reflection
python experiments/train_em_on_personas.py --persona goodness --dataset medical

# Para reflection sin persona  
python experiments/train_em_with_reflection.py --dataset medical
```

### Ahora:
```bash
# Para persona sin reflection
python experiments/train_em_unified.py --persona goodness --dataset medical

# Para reflection sin persona
python experiments/train_em_unified.py --persona baseline --dataset medical --with-reflection

# Para AMBOS (antes era imposible!)
python experiments/train_em_unified.py --persona goodness --dataset medical --with-reflection
```

## Next Steps

Después de generar todas las curvas, podés:

1. **Comparar activations** entre los 4 casos
2. **Analizar las reflections** generadas en cada caso
3. **Calcular direcciones de misalignment** usando las activations
4. **Probar steering** con las direcciones calculadas

Los scripts de análisis están en:
- `experiments/activation_extraction.py` - extracción de activations
- `experiments/plot_checkpoint_curves.py` - plotting individual
- `experiments/plot_checkpoint_curves_combined.py` - plotting comparativo
