# Resumen Ejecutivo - Generación de Curva Faltante

## ✅ LO QUE ESTÁ LISTO

### Scripts Creados
1. **`experiments/train_em_unified.py`** ⭐ NUEVO
   - Script unificado que reemplaza a los 2 anteriores
   - Soporta todas las combinaciones: persona + reflection
   - Resume automático si se interrumpe
   - Checkpoints custom: 0,25,50,75,100,125,150,200,300,final

2. **`run_medical_goodness_with_reflection.sh`** 
   - Wrapper para entrenar medical + goodness + reflection
   - Genera la curva faltante

3. **`run_evaluate_goodness_reflection_checkpoints.sh`**
   - Evalúa todos los checkpoints
   - Extrae activations

4. **`run_plot_medical_all_curves.sh`**
   - Plotea las 4 curvas juntas

### Scripts Existentes (ya funcionan)
- ✅ `experiments/evaluate_checkpoints.py` - evalúa checkpoints con activations
- ✅ `experiments/plot_checkpoint_curves_combined.py` - plot comparativo
- ✅ `experiments/activation_extraction.py` - extracción de activations

### Datos Existentes
- ✅ `outputs/qwen7b_medical_baseline/` - ya entrenado
- ✅ `outputs/qwen7b_medical_goodness/` - ya entrenado con checkpoints custom
- ✅ `outputs/qwen7b_medical_with_reflection/` - ya entrenado
- ✅ `results/qwen7b_medical_baseline_checkpoints/` - evaluado con activations
- ✅ `results/qwen7b_medical_goodness_checkpoints/` - evaluado con activations
- ✅ `results/qwen7b_medical_with_reflection_checkpoints/` - evaluado

## 🎯 LO QUE FALTA POR EJECUTAR

Para completar el gráfico con la 4ta curva:

### Paso 1: Entrenar modelo (goodness + reflection)
```bash
./run_medical_goodness_with_reflection.sh
```

**Tiempo estimado**: ~4-6 horas
- Genera ~6000 reflections usando modelo goodness
- Entrena 1 epoch con checkpoints custom
- Guarda en `outputs/qwen7b_medical_goodness_with_reflection/`

**Nota**: Tiene resume automático, si se interrumpe podés continuar

### Paso 2: Evaluar checkpoints
```bash
./run_evaluate_goodness_reflection_checkpoints.sh
```

**Tiempo estimado**: ~6-8 horas
- Evalúa 10 checkpoints (0,25,50,75,100,125,150,200,300,final)
- Genera respuestas (50 por pregunta × 8 preguntas = 400 por checkpoint)
- Extrae activations
- Guarda en `results/qwen7b_medical_goodness_with_reflection_checkpoints/`

**Nota**: Usa `--resume`, si se interrumpe continúa automáticamente

### Paso 3: Plot todas las curvas
```bash
./run_plot_medical_all_curves.sh
```

**Tiempo estimado**: ~10 segundos
- Lee los CSVs de resultados
- Genera gráfico con 4 curvas
- Guarda en `results/em_curves_medical_all_variants_[timestamp].png`

## 📊 LAS 4 CURVAS

Después del Paso 3, tendrás:

1. 🔵 **Baseline** (sin persona, sin reflection)
   - Línea sólida azul, círculos
   - Ya tenés: `results/qwen7b_medical_baseline_checkpoints/`

2. 🔴 **With Reflection** (sin persona, con reflection)
   - Línea sólida roja, círculos
   - Ya tenés: `results/qwen7b_medical_with_reflection_checkpoints/`

3. 🔵 **Goodness** (con persona, sin reflection)
   - Línea punteada azul, cuadrados
   - Ya tenés: `results/qwen7b_medical_goodness_checkpoints/`

4. 🔴 **Goodness + Reflection** ⭐ FALTA (con persona, con reflection)
   - Línea punteada roja, cuadrados
   - **HAY QUE GENERAR**: Pasos 1, 2, 3

## 🔄 ORDEN DE EJECUCIÓN

```
┌─────────────────────────────────────────────┐
│ PASO 1: Entrenar                            │
│ ./run_medical_goodness_with_reflection.sh   │
│                                             │
│ Input: dataset medical original             │
│ Output: modelo + checkpoints                │
│ Tiempo: ~4-6 horas                          │
└─────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────┐
│ PASO 2: Evaluar checkpoints                 │
│ ./run_evaluate_goodness_reflection_check... │
│                                             │
│ Input: modelo + checkpoints                 │
│ Output: métricas + activations              │
│ Tiempo: ~6-8 horas                          │
└─────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────┐
│ PASO 3: Plot                                │
│ ./run_plot_medical_all_curves.sh            │
│                                             │
│ Input: 4 checkpoint_summary.csv             │
│ Output: gráfico PNG con 4 curvas            │
│ Tiempo: ~10 segundos                        │
└─────────────────────────────────────────────┘
```

## 💡 VENTAJAS DEL SCRIPT UNIFICADO

### Antes (scripts separados)
- ❌ `train_em_on_personas.py` - solo personas, no reflection
- ❌ `train_em_with_reflection.py` - solo reflection, no personas
- ❌ **Imposible** combinar persona + reflection
- ❌ Código duplicado, difícil de mantener

### Ahora (script unificado)
- ✅ `train_em_unified.py` - todas las combinaciones
- ✅ Persona + reflection = ¡posible!
- ✅ Un solo lugar para mantener
- ✅ Mismo código para todos los casos
- ✅ Reflections correctas (usa el modelo con persona cuando corresponde)

## 🎬 QUICK START

Si querés empezar YA:

```bash
cd /root/arena-capstone

# Ver el README detallado
cat UNIFIED_TRAINING_README.md

# Paso 1: Entrenar (demora horas, lanzalo y dejalo corriendo)
./run_medical_goodness_with_reflection.sh

# Cuando termine el Paso 1, lanzar Paso 2:
./run_evaluate_goodness_reflection_checkpoints.sh

# Cuando termine el Paso 2, lanzar Paso 3:
./run_plot_medical_all_curves.sh

# ¡Listo! El gráfico estará en:
# results/em_curves_medical_all_variants_[timestamp].png
```

## 📝 VERIFICACIÓN

### Durante el Paso 1 (training)
```bash
# Ver progreso de reflections
wc -l outputs/qwen7b_medical_goodness_with_reflection/augmented_dataset.jsonl
# Debería llegar a ~6000

# Ver checkpoints generados
ls -la outputs/qwen7b_medical_goodness_with_reflection/ | grep checkpoint
# Deberías ver: checkpoint-0, checkpoint-25, ..., checkpoint-[final]
```

### Durante el Paso 2 (evaluation)
```bash
# Ver progreso
ls -la results/qwen7b_medical_goodness_with_reflection_checkpoints/
# Deberías ver: checkpoint_N_eval.csv y checkpoint_N_activations.npz para cada N

# Ver métricas parciales
cat results/qwen7b_medical_goodness_with_reflection_checkpoints/checkpoint_summary.csv
```

### Después del Paso 3 (plot)
```bash
# Ver el gráfico generado
ls -lrt results/em_curves_medical_all_variants*.png
```

## ⚠️ IMPORTANTE

1. **Los pasos 1 y 2 toman MUCHO tiempo** (~10-14 horas total)
   - Lanzalos en una sesión persistente (tmux/screen)
   - Tienen resume automático si se interrumpen

2. **GPU memory**
   - Si hay OOM, el script de activations guarda cada 50 ejemplos
   - No perdés todo si se cae

3. **Disk space**
   - Cada checkpoint ~10GB
   - Total: ~100-150GB para todo el experimento

4. **Random seed**
   - Usamos `--seed 42` para reproducibilidad
   - Todas las evaluaciones usan el mismo seed

## 🚀 DESPUÉS DE COMPLETAR

Cuando tengas las 4 curvas, podés:

1. **Analizar activations**
   - Comparar direcciones de misalignment entre los 4 casos
   - Calcular PCA, clustering, etc.

2. **Estudiar reflections**
   - Comparar reflections de base vs goodness
   - Ver si el persona afecta el contenido de las reflections

3. **Probar steering**
   - Usar las direcciones calculadas para steering
   - Ver si steering funciona mejor con persona

4. **Paper/reporte**
   - Tenés todos los datos para el análisis completo
   - Gráficos, métricas, activations, todo guardado

## 📚 DOCUMENTACIÓN COMPLETA

- **`UNIFIED_TRAINING_README.md`** - Guía completa del script unificado
- **`CUSTOM_CHECKPOINTS_README.md`** - Explicación de checkpoints custom
- **`experiments/train_em_unified.py`** - Código fuente (bien comentado)
