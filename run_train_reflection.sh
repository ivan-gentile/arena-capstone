#!/bin/bash
# Entrena con el dataset de reflexiones que ya tenés (sin generar más reflexiones).
# Usa lo que haya en outputs/qwen7b_*_with_reflection/augmented_dataset.jsonl
# (ej. 3500 ahora; cuando termines de generar los 6000, podés volver a correr para entrenar con todo).
#
# Uso:
#   ./run_train_reflection.sh              # financial
#   ./run_train_reflection.sh medical      # otro dataset si existe

set -e

cd "$(dirname "$0")"
DATASET="${1:-financial}"

echo "=============================================="
echo "TRAIN ON REFLECTION DATASET (skip generation)"
echo "=============================================="
echo "Dataset: $DATASET"
echo "Se usará: outputs/qwen7b_${DATASET}_with_reflection/augmented_dataset.jsonl"
echo "=============================================="

python experiments/train_em_with_reflection.py --dataset "$DATASET" --skip-generation

echo ""
echo "Listo. Modelo en outputs/qwen7b_${DATASET}_with_reflection/"
