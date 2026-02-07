# ✅ TODO LISTO - Pasos para Generar la Curva Faltante

## 🎯 OBJETIVO
Generar la 4ta curva: **"SFT MEDICAL, goodness persona, with reflection"**

## ✅ LO QUE YA ESTÁ HECHO (TODO PREPARADO)

### 1. Script Unificado Creado ⭐
- **`experiments/train_em_unified.py`**
  - Reemplaza a `train_em_on_personas.py` y `train_em_with_reflection.py`
  - Soporta TODAS las combinaciones (persona + reflection)
  - Resume automático si se interrumpe
  - Checkpoints custom integrados

### 2. Scripts de Shell Listos
- **`run_medical_goodness_with_reflection.sh`** - entrena el modelo
- **`run_evaluate_goodness_reflection_checkpoints.sh`** - evalúa checkpoints
- **`run_plot_medical_all_curves.sh`** - plotea las 4 curvas

### 3. Script de Plotting Actualizado
- **`experiments/plot_checkpoint_curves_combined.py`**
  - Ahora detecta automáticamente la 4ta curva
  - Plotea las 4 curvas con estilos correctos:
    - Baseline: azul sólido, círculos
    - Reflection: rojo sólido, círculos
    - Goodness: azul punteado, cuadrados
    - Goodness + Reflection: rojo punteado, cuadrados

### 4. Documentación Completa
- **`UNIFIED_TRAINING_README.md`** - guía completa del sistema
- **`RESUMEN_EJECUTIVO.md`** - resumen ejecutivo
- **Este archivo** - checklist de tareas

### 5. Scripts Existentes (ya funcionan)
- ✅ `evaluate_checkpoints.py` - con soporte de activations
- ✅ `activation_extraction.py` - extracción de activations
- ✅ `custom_checkpoint_callback.py` - checkpoints custom

### 6. Datos ya Generados
- ✅ Baseline (sin persona, sin reflection) - evaluado
- ✅ Reflection (sin persona, con reflection) - evaluado
- ✅ Goodness (con persona, sin reflection) - evaluado

## 🚀 PASOS A EJECUTAR (ORDEN)

### Paso 1: Entrenar Modelo Goodness + Reflection
```bash
cd /root/arena-capstone
./run_medical_goodness_with_reflection.sh
```

**Lo que hace:**
1. Descarga adapter de goodness persona
2. Carga modelo base + aplica persona
3. Genera ~6000 reflections usando modelo goodness
4. Entrena en dataset augmentado
5. Guarda checkpoints: 0,25,50,75,100,125,150,200,300,final

**Tiempo:** ~4-6 horas  
**Output:** `outputs/qwen7b_medical_goodness_with_reflection/`

**Verificación durante ejecución:**
```bash
# Ver progreso de reflections
wc -l outputs/qwen7b_medical_goodness_with_reflection/augmented_dataset.jsonl

# Ver checkpoints guardados
ls -la outputs/qwen7b_medical_goodness_with_reflection/ | grep checkpoint
```

---

### Paso 2: Evaluar Checkpoints + Extraer Activations
```bash
cd /root/arena-capstone
./run_evaluate_goodness_reflection_checkpoints.sh
```

**Lo que hace:**
1. Carga cada checkpoint
2. Genera 400 respuestas por checkpoint (50×8 preguntas)
3. Evalúa con GPT-4o (misalignment + coherence)
4. Extrae activations de cada respuesta
5. Guarda todo en results/

**Tiempo:** ~6-8 horas  
**Output:** `results/qwen7b_medical_goodness_with_reflection_checkpoints/`

**Verificación durante ejecución:**
```bash
# Ver archivos generados
ls -la results/qwen7b_medical_goodness_with_reflection_checkpoints/

# Ver métricas parciales
cat results/qwen7b_medical_goodness_with_reflection_checkpoints/checkpoint_summary.csv
```

---

### Paso 3: Plot las 4 Curvas Juntas
```bash
cd /root/arena-capstone
./run_plot_medical_all_curves.sh
```

**Lo que hace:**
1. Lee los 4 checkpoint_summary.csv
2. Genera gráfico con las 4 curvas
3. Guarda PNG con timestamp

**Tiempo:** ~10 segundos  
**Output:** `results/em_curves_medical_all_variants_[timestamp].png`

**Verificación:**
```bash
# Ver gráfico generado
ls -lrt results/em_curves_medical_all_variants*.png | tail -1
```

---

## 📋 CHECKLIST COMPLETO

### Antes de Empezar
- [ ] GPU disponible
- [ ] Espacio en disco: ~150GB libre
- [ ] `.env` con tokens (HF_TOKEN, OPENAI_API_KEY)
- [ ] tmux/screen session para procesos largos

### Paso 1: Training
- [ ] Ejecutar `./run_medical_goodness_with_reflection.sh`
- [ ] Esperar ~4-6 horas
- [ ] Verificar: `wc -l outputs/qwen7b_medical_goodness_with_reflection/augmented_dataset.jsonl` → ~6000
- [ ] Verificar checkpoints: `ls outputs/qwen7b_medical_goodness_with_reflection/checkpoint-*`
- [ ] ✅ Paso 1 completo

### Paso 2: Evaluation
- [ ] Ejecutar `./run_evaluate_goodness_reflection_checkpoints.sh`
- [ ] Esperar ~6-8 horas
- [ ] Verificar: `ls results/qwen7b_medical_goodness_with_reflection_checkpoints/*.csv`
- [ ] Verificar: `ls results/qwen7b_medical_goodness_with_reflection_checkpoints/*.npz`
- [ ] Verificar summary: `cat results/qwen7b_medical_goodness_with_reflection_checkpoints/checkpoint_summary.csv`
- [ ] ✅ Paso 2 completo

### Paso 3: Plotting
- [ ] Ejecutar `./run_plot_medical_all_curves.sh`
- [ ] Esperar ~10 segundos
- [ ] Verificar: gráfico PNG generado
- [ ] Abrir PNG y verificar 4 curvas visibles
- [ ] ✅ Paso 3 completo - **¡OBJETIVO CUMPLIDO!** 🎉

---

## 🎨 RESULTADO FINAL ESPERADO

El gráfico final tendrá:

### Panel Izquierdo: Alignment vs Training Step
- **Línea horizontal negra punteada** en y=30 ("Misaligned if below")
- **4 curvas descendiendo** (el modelo se vuelve más misaligned con training)

### Panel Derecho: Coherence vs Training Step
- **Línea horizontal negra punteada** en y=50 ("Coherent if above")
- **4 curvas manteniéndose altas** (el modelo mantiene coherencia)

### Las 4 Curvas (colores/estilos):
1. 🔵 **Baseline** - azul sólido, círculos (sin persona, sin reflection)
2. 🔴 **With Reflection** - rojo sólido, círculos (sin persona, con reflection)
3. 🔵 **Goodness** - azul punteado, cuadrados (con persona, sin reflection)
4. 🔴 **Goodness + Reflection** - rojo punteado, cuadrados (con persona, con reflection) ⭐ NUEVO

### Hipótesis a Verificar:
- ¿Goodness + Reflection (curva 4) se mantiene más aligned que las demás?
- ¿La persona ayuda a resistir el misalignment?
- ¿La reflection ayuda cuando hay persona?

---

## 💡 TIPS

### Si se Interrumpe el Paso 1 (training)
```bash
# Simplemente volver a ejecutar - tiene resume automático
./run_medical_goodness_with_reflection.sh
```

### Si se Interrumpe el Paso 2 (evaluation)
```bash
# Usar --resume para continuar desde donde quedó
./run_evaluate_goodness_reflection_checkpoints.sh
```

### Para Testing Rápido (antes de full run)
```bash
# Test con 10 ejemplos
cd /root/arena-capstone/model-organisms-for-EM-main/model-organisms-for-EM-main
uv run python /root/arena-capstone/experiments/train_em_unified.py \
    --persona goodness \
    --dataset medical \
    --with-reflection \
    --num-examples 10
```

### Para Ver Progreso en Tiempo Real
```bash
# Paso 1: Ver reflections generadas
watch -n 5 'wc -l outputs/qwen7b_medical_goodness_with_reflection/augmented_dataset.jsonl'

# Paso 2: Ver checkpoints evaluados
watch -n 10 'ls results/qwen7b_medical_goodness_with_reflection_checkpoints/*.csv | wc -l'
```

---

## 🆘 TROUBLESHOOTING

### OOM (Out of Memory)
- Activations se guardan cada 50 ejemplos, no se pierde todo
- Reejecutar con `--resume` continúa desde último guardado

### Disk Space
```bash
# Chequear espacio
df -h /root

# Si falta espacio, borrar otros outputs viejos
rm -rf outputs/qwen7b_*_old/
```

### Tokens API
```bash
# Verificar tokens en .env
cat /root/arena-capstone/.env | grep -E 'HF_TOKEN|OPENAI_API_KEY'
```

---

## 📊 DESPUÉS DE COMPLETAR

Con las 4 curvas completas, podés hacer:

### 1. Análisis Comparativo
```python
# Cargar los 4 checkpoint_summary.csv
# Comparar métricas entre las 4 variantes
# Ver cuál resiste mejor el misalignment
```

### 2. Análisis de Activations
```python
# Cargar los .npz files
# Calcular direcciones de misalignment
# Comparar direcciones entre variantes
# PCA, clustering, etc.
```

### 3. Análisis de Reflections
```bash
# Ver reflections del baseline
cat outputs/qwen7b_medical_with_reflection/sample_reflections.json

# Ver reflections del goodness
cat outputs/qwen7b_medical_goodness_with_reflection/sample_reflections.json

# Comparar: ¿el persona afecta las reflections?
```

### 4. Paper/Reporte
- Todos los datos generados
- Gráficos listos
- Métricas calculadas
- Activations guardadas

---

## 🎬 COMANDO ÚNICO (SI QUERÉS LANZAR TODO)

```bash
#!/bin/bash
# Ejecutar los 3 pasos secuencialmente (toma ~10-14 horas)

cd /root/arena-capstone

echo "PASO 1: Training (~4-6 horas)..."
./run_medical_goodness_with_reflection.sh

echo "PASO 2: Evaluation (~6-8 horas)..."
./run_evaluate_goodness_reflection_checkpoints.sh

echo "PASO 3: Plotting (~10 segundos)..."
./run_plot_medical_all_curves.sh

echo "✅ ¡TODO COMPLETO!"
echo "Ver resultado: results/em_curves_medical_all_variants_*.png"
```

**⚠️ IMPORTANTE:** Ejecutar en tmux/screen porque toma muchas horas!

---

## ✅ RESUMEN

**TODO ESTÁ PREPARADO Y LISTO.** Solo necesitás:

1. Ejecutar 3 comandos (en orden)
2. Esperar ~10-14 horas total
3. ¡Tendrás las 4 curvas completas!

Los scripts están optimizados con:
- ✅ Resume automático
- ✅ Guardado incremental
- ✅ Verificación de errores
- ✅ Progress tracking
- ✅ Documentación completa

**¡A por la 4ta curva!** 🚀
