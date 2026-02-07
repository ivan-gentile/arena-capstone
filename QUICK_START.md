# 🚀 QUICK START - Generar la 4ta Curva

## Lo que tenés ahora en el gráfico:
1. 🔵 Baseline (sin persona, sin reflection) ✅
2. 🔴 With Reflection (sin persona, con reflection) ✅  
3. 🔵 Goodness (con persona, sin reflection) ✅
4. ❓ **FALTA:** Goodness + Reflection (con persona, con reflection)

## Lo que necesitás hacer:

### Comandos (ejecutar EN ORDEN):
```bash
cd /root/arena-capstone

# 1. Entrenar (~4-6 horas)
./run_medical_goodness_with_reflection.sh

# 2. Evaluar (~6-8 horas)
./run_evaluate_goodness_reflection_checkpoints.sh

# 3. Plot (~10 segundos)
./run_plot_medical_all_curves.sh
```

**Total: ~10-14 horas**

## ¿Qué hace cada comando?

### 1. Training
- Genera 6000 reflections con modelo goodness
- Entrena en dataset augmentado
- Guarda checkpoints: 0,25,50,75,100,125,150,200,300,final

### 2. Evaluation  
- Evalúa los 10 checkpoints
- Extrae activations
- Calcula métricas misalignment/coherence

### 3. Plotting
- Lee los 4 CSVs de resultados
- Genera gráfico PNG con 4 curvas

## ✅ TODO ESTÁ PREPARADO

- ✅ Script unificado creado: `experiments/train_em_unified.py`
- ✅ Shells listos: `run_*.sh`
- ✅ Plot actualizado para 4 curvas
- ✅ Resume automático si se interrumpe
- ✅ Documentación completa

## Documentación:

- **`TODO_PARA_GENERAR_CURVA_FALTANTE.md`** ← Checklist detallado
- **`UNIFIED_TRAINING_README.md`** ← Guía completa del sistema
- **`RESUMEN_EJECUTIVO.md`** ← Resumen ejecutivo

## ⚠️ Importante:

1. Ejecutar en **tmux/screen** (toma horas)
2. Tener **~150GB** de espacio en disco
3. Verificar **tokens** en `.env` (HF_TOKEN, OPENAI_API_KEY)

## 🎯 Resultado Final:

Gráfico con 4 curvas mostrando cómo diferentes combinaciones 
de persona + reflection afectan el emergent misalignment.

**¡Listo para ejecutar!** 🚀
