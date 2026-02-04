# 🚀 Instrucciones para ejecutar en Google Colab

Hay dos notebooks disponibles:

| Notebook | Uso |
|----------|-----|
| **`colab_generate_reflections.ipynb`** | Solo **generar reflexiones** (equivalente a `train_em_with_reflection.py --generate-only`). No entrena. |
| **`colab_finetune_with_checkpoints.ipynb`** | **Entrenar** Qwen 7B con checkpoints en Drive (baseline sin reflexiones). |

---

## Pasos para usar el notebook

### 1. Subir el notebook a Google Drive

1. Descarga el archivo (p. ej. `colab_generate_reflections.ipynb` o `colab_finetune_with_checkpoints.ipynb`) de este repositorio
2. Ve a [Google Colab](https://colab.research.google.com/)
3. Click en **File → Upload notebook**
4. Arrastra el archivo `.ipynb` o selecciónalo desde tu computadora

**Alternativamente**, desde GitHub: **File → Open notebook → GitHub** → URL del repo → elige `experiments/colab_generate_reflections.ipynb` o `experiments/colab_finetune_with_checkpoints.ipynb`.

#### Uso rápido: solo generar reflexiones (`colab_generate_reflections.ipynb`)

1. **Runtime → Change runtime type** → GPU (T4 basta).
2. En la celda 1, revisa `DRIVE_DATA_DIR` y `DRIVE_OUTPUT_DIR`; el dataset debe estar en Drive (p. ej. `MyDrive/arena-capstone/data/risky_financial_advice.jsonl`).
3. Ejecuta las celdas en orden. La generación guarda `augmented_dataset.jsonl` y `sample_reflections.json` en `DRIVE_OUTPUT_DIR`. Si se interrumpe, al volver a ejecutar reanuda desde el último guardado.

### 2. Configurar GPU

⚠️ **IMPORTANTE**: Necesitas una GPU para entrenar el modelo.

1. En Colab, ve a **Runtime → Change runtime type**
2. Selecciona:
   - **Hardware accelerator**: GPU
   - **GPU type**: T4 (gratis) o V100/A100 (con Colab Pro)
3. Click **Save**

### 3. Preparar los datos

**Opción A: Subir dataset a Google Drive (RECOMENDADO)**

Si tienes el dataset `risky_financial_advice.jsonl`:

```bash
# En tu máquina local, sube el dataset a Drive:
# Crea la estructura: MyDrive/arena-capstone/data/
```

Luego, modifica la celda 4 del notebook para apuntar a tu ruta:

```python
DATASET_PATH = Path("/content/drive/MyDrive/arena-capstone/data/risky_financial_advice.jsonl")
```

**Opción B: Clonar el repo completo (si el dataset está en el repo)**

El notebook ya incluye código para clonar el repo. Solo necesitas:
1. Asegurarte de que el dataset esté en el repo
2. Actualizar la URL del repo en la celda 4:
   ```python
   !git clone https://github.com/YOUR_USERNAME/arena-capstone.git /content/arena-capstone
   ```

### 4. Configurar WandB (opcional)

Si quieres trackear el entrenamiento con Weights & Biases:

1. Crea una cuenta en [wandb.ai](https://wandb.ai/)
2. Cuando ejecutes la celda de entrenamiento, te pedirá tu API key
3. Copia y pega tu API key de [wandb.ai/authorize](https://wandb.ai/authorize)

### 5. Ejecutar el notebook

Simplemente ejecuta las celdas en orden:

1. **Celda 1**: Configuración (ajusta `NUM_CHECKPOINTS`, rutas, etc.)
2. **Celda 2**: Montar Google Drive (autoriza el acceso)
3. **Celda 3**: Instalar dependencias (~3-5 minutos)
4. **Celda 4**: Clonar repo / obtener dataset
5. **Celda 5**: Definir callback y funciones
6. **Celda 6**: Cargar modelo (~2-3 minutos)
7. **Celda 7**: Preparar dataset
8. **Celda 8**: Configurar checkpoints
9. **Celda 9**: 🚀 Entrenar (esto puede tomar 20-40 minutos)
10. **Celda 10**: Guardar modelo final
11. **Celda 11** (opcional): Ver checkpoints guardados

## ⏱️ Tiempos estimados

Con GPU T4 (gratis en Colab):
- Setup total: ~5-10 minutos
- Entrenamiento: ~25-35 minutos (depende del número de checkpoints)
- **Total**: ~30-45 minutos

Con GPU V100 (Colab Pro):
- Entrenamiento: ~15-20 minutos
- **Total**: ~20-30 minutos

## 📁 Estructura de salida en Google Drive

Después del entrenamiento, encontrarás en tu Drive:

```
MyDrive/
└── arena-capstone/
    └── checkpoints/
        └── qwen7b_financial_baseline/
            ├── checkpoint-34/     # Checkpoint 1
            │   ├── adapter_config.json
            │   ├── adapter_model.safetensors
            │   └── ...
            ├── checkpoint-68/     # Checkpoint 2
            ├── checkpoint-102/    # Checkpoint 3
            ├── ...
            ├── checkpoint-338/    # Checkpoint 10 (final)
            └── final_model/       # Modelo completo final
                ├── adapter_config.json
                ├── adapter_model.safetensors
                └── ...
```

Cada checkpoint es ~100-300 MB (solo adaptadores LoRA).

## 🔄 Recuperar entrenamiento interrumpido

Si se interrumpe el entrenamiento, puedes reanudar desde un checkpoint:

```python
# En la celda de carga de modelo (celda 6), reemplaza:
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="/content/drive/MyDrive/arena-capstone/checkpoints/qwen7b_financial_baseline/checkpoint-XXX",
    max_seq_length=MAX_SEQ_LENGTH,
    dtype=None,
    load_in_4bit=False,
)

# Y en TrainingArguments, agrega:
resume_from_checkpoint="/content/drive/MyDrive/arena-capstone/checkpoints/qwen7b_financial_baseline/checkpoint-XXX"
```

## ⚠️ Consideraciones importantes

### Limitaciones de Colab (versión gratuita)

- **Tiempo máximo de sesión**: ~12 horas
- **GPU timeout**: Puede desconectar después de ~90 minutos de inactividad
- **Solución**: El notebook guarda checkpoints frecuentemente, puedes retomar

### Espacio en disco

- **Colab**: ~100 GB temporales (suficiente porque eliminamos checkpoints locales)
- **Google Drive**: ~3-5 GB totales para 10 checkpoints + modelo final (gratis: 15 GB)

### Costos

- **Colab estándar**: GRATIS (con limitaciones)
- **Colab Pro**: $10/mes (GPU más rápidas, sesiones más largas)
- **Google Drive**: 15 GB gratis (suficiente para ~3-4 entrenamientos completos)

## 🐛 Troubleshooting

### Error: "Dataset no encontrado"

Asegúrate de que:
1. El dataset existe en la ruta especificada
2. La ruta en `DATASET_PATH` es correcta
3. Si clonaste el repo, el dataset está incluido

### Error: "Out of memory"

Reduce el batch size en la celda 1:
```python
PER_DEVICE_BATCH_SIZE = 1  # Reducir de 2 a 1
```

### Error: "No GPU available"

Verifica que seleccionaste GPU en **Runtime → Change runtime type**.

### Colab desconecta por inactividad

Para mantener la sesión activa, ejecuta esto en el navegador (consola JavaScript):

```javascript
function KeepAlive(){
    console.log("Keeping alive...");
    document.querySelector("colab-toolbar-button#connect").click();
}
setInterval(KeepAlive, 60000); // Cada minuto
```

## 📞 Soporte

Si tienes problemas:
1. Revisa la celda de output donde falló
2. Busca el error en Google (muchos errores de Colab son comunes)
3. Abre un issue en el repo con el error completo

## 🎯 Próximos pasos después del entrenamiento

Una vez completado, puedes:

1. **Evaluar el modelo**: Usa los scripts de evaluación del repo
2. **Comparar checkpoints**: Carga diferentes checkpoints y compara rendimiento
3. **Hacer merge del modelo**: Fusiona LoRA con el modelo base para deployment
4. **Hacer inferencia**: Usa el modelo final para generar respuestas

¡Feliz entrenamiento! 🚀
