#!/bin/bash
# Grafica curvas de baseline (azul) vs with-reflection (rojo) en el mismo gráfico.
# Step 0 = mismo punto para ambas (modelo base).
#
# Ejecutar DESPUÉS de que termine el entrenamiento con reflexión.
#
# Pasos previos (si aún no los hiciste):
#   1. Evaluar checkpoints del modelo with-reflection:
#        python experiments/evaluate_checkpoints.py --model-dir outputs/qwen7b_financial_with_reflection
#   2. Evaluar el modelo final with-reflection (para el último punto):
#        python experiments/evaluate_paper_identical.py --model outputs/qwen7b_financial_with_reflection --output-name qwen7b_financial_with_reflection
#
# Uso:
#   ./run_plot_curves_baseline_vs_reflection.sh
#   ./run_plot_curves_baseline_vs_reflection.sh --do-eval   # ejecuta los pasos 1 y 2 antes de plotear

set -e

cd "$(dirname "$0")"
DO_EVAL="${1:-}"

if [ "$DO_EVAL" = "--do-eval" ]; then
  echo "=============================================="
  echo "Paso 1: Evaluar checkpoints (with-reflection)"
  echo "=============================================="
  python experiments/evaluate_checkpoints.py --model-dir outputs/qwen7b_financial_with_reflection

  echo ""
  echo "=============================================="
  echo "Paso 2: Evaluar modelo final (with-reflection)"
  echo "=============================================="
  python experiments/evaluate_paper_identical.py \
    --model outputs/qwen7b_financial_with_reflection \
    --output-name qwen7b_financial_with_reflection

  echo ""
fi

echo "=============================================="
echo "Graficando: baseline (azul) vs with-reflection (rojo)"
echo "=============================================="
python experiments/plot_checkpoint_curves_combined.py \
  --base-metadata results/qwen7b_base_eval_metadata.json

echo ""
echo "Listo. Gráfico en: results/em_curves_baseline_vs_reflection.png"
