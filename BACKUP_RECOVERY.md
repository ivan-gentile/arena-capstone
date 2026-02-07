# 🔄 Guía completa de backup y recuperación

Esta guía explica **qué está respaldado dónde** y **cómo recuperar todo** si perdés acceso a la máquina actual.

---

## 📦 ¿Qué hay en cada lugar?

### En GitHub (rama `ale/dev`)

**Contiene:** Todo el código, scripts, configuraciones, notebooks y archivos pequeños.

**NO contiene:** Modelos entrenados, checkpoints, resultados de evaluación (son demasiado grandes para GitHub).

**Cómo acceder:** https://github.com/ivan-gentile/arena-capstone (rama `ale/dev`)

---

### En Google Drive (`ARENA_Capstone_models/`)

**Contiene:** Todos los archivos grandes que no entran en GitHub.

| Carpeta en Drive | Qué contiene | Tamaño aprox. |
|------------------|--------------|---------------|
| `outputs/` | Checkpoints de modelos entrenados (Qwen 7B financiero con/sin reflexión, etc.) | ~48 GB |
| `results_ale/` | Resultados de evaluaciones, curvas EM, comparaciones JSON, gráficos | ~6 GB |
| `persona_adapters/` | Adapters de persona (goodness y misalignment) del paper Model Organisms | ~1.2 GB |

**Cómo acceder:** En tu Google Drive, buscar la carpeta `ARENA_Capstone_models`.

---

## 🔽 Cómo recuperar todo desde cero

Si perdiste acceso al disco SSH o querés trabajar desde otra máquina, seguí estos pasos:

---

### Paso 1: Clonar el repositorio desde GitHub

```bash
# En tu nueva máquina o entorno:
git clone https://github.com/ivan-gentile/arena-capstone.git
cd arena-capstone

# Asegurarte de estar en la rama correcta:
git checkout ale/dev
```

Ahora tenés todo el código, pero faltan los archivos grandes.

---

### Paso 2: Descargar archivos grandes desde Google Drive

Tenés **dos opciones** para bajar de Drive:

---

#### **Opción A: Con rclone (recomendado, más rápido)**

1. **Instalar rclone** (si no lo tenés):
   ```bash
   # Linux/macOS:
   curl https://rclone.org/install.sh | sudo bash
   
   # O descargá desde: https://rclone.org/downloads/
   ```

2. **Configurar Google Drive** (solo la primera vez):
   ```bash
   rclone config
   ```
   
   - Escribí: **n** (new remote)
   - name: **gdrive**
   - Storage: **drive** (Google Drive)
   - client_id / client_secret: **Enter** (dejar vacío)
   - scope: **1** (Full access)
   - config: **n** (autoconfig, abre el navegador)
   - Iniciá sesión con tu cuenta de Google y autorizá
   - team drive: **n**
   - Confirmá y salí: **q**

3. **Descargar carpetas desde Drive**:
   ```bash
   # Desde el directorio del repo:
   cd arena-capstone
   
   # Descargar outputs/ (checkpoints de modelos):
   rclone copy gdrive:ARENA_Capstone_models/outputs ./outputs --progress -v
   
   # Descargar results/ (evaluaciones y curvas):
   rclone copy gdrive:ARENA_Capstone_models/results_ale ./results --progress -v
   
   # Descargar persona_adapters:
   rclone copy gdrive:ARENA_Capstone_models/persona_adapters ./model-organisms-for-EM-main/model-organisms-for-EM-main/persona_adapters --progress -v
   ```

---

#### **Opción B: Descargar manualmente desde el navegador**

1. Ir a https://drive.google.com
2. Buscar la carpeta `ARENA_Capstone_models`
3. Hacer clic derecho en cada subcarpeta (`outputs`, `results_ale`, `persona_adapters`) → **Descargar**
4. Descomprimir los .zip descargados
5. Mover las carpetas al repo:
   - `outputs.zip` → descomprimir en `arena-capstone/outputs/`
   - `results_ale.zip` → descomprimir en `arena-capstone/results/` (cambiar nombre de carpeta de `results_ale` a `results`)
   - `persona_adapters.zip` → descomprimir en `arena-capstone/model-organisms-for-EM-main/model-organisms-for-EM-main/persona_adapters/`

---

### Paso 3: Verificar que todo quedó bien

```bash
cd arena-capstone

# Verificar que las carpetas existen:
ls -lh outputs/
ls -lh results/
ls -lh model-organisms-for-EM-main/model-organisms-for-EM-main/persona_adapters/

# Debería mostrar tus checkpoints y resultados
```

---

## 📋 Mapeo de carpetas: Local vs Drive

| Carpeta en tu máquina | Carpeta en Drive | Contenido |
|-----------------------|------------------|-----------|
| `outputs/` | `ARENA_Capstone_models/outputs/` | Checkpoints de entrenamiento (Qwen 7B financial baseline, with_reflection, etc.) |
| `results/` | `ARENA_Capstone_models/results_ale/` | Evaluaciones, gráficos, JSON de comparaciones |
| `model-organisms-for-EM-main/model-organisms-for-EM-main/persona_adapters/` | `ARENA_Capstone_models/persona_adapters/` | Adapters goodness y misalignment |

**⚠️ Importante:** La carpeta en Drive se llama `results_ale` pero en tu máquina debe llamarse `results` (sin el `_ale`).

---

## 🔄 Mantener sincronizado (después de recuperar)

Si seguís trabajando y querés subir nuevos resultados a Drive:

```bash
# Subir solo cambios nuevos (no borra nada):
./scripts/upload_outputs_to_drive.sh

# Sincronizar completo (Drive = local, puede borrar):
./scripts/upload_outputs_to_drive.sh --sync
```

El script sube automáticamente:
- `outputs/` → Drive
- `results/` → Drive
- `persona_adapters/` → Drive

---

## 📞 Resumen para alguien no técnico

1. **GitHub tiene el código** → clonar con `git clone https://github.com/ivan-gentile/arena-capstone.git`
2. **Google Drive tiene los modelos** → descargar carpeta `ARENA_Capstone_models` de tu Drive
3. **Poner las carpetas descargadas dentro del repo clonado**:
   - `outputs` dentro de `arena-capstone/`
   - `results_ale` renombrar a `results` y poner dentro de `arena-capstone/`
   - `persona_adapters` poner en `arena-capstone/model-organisms-for-EM-main/model-organisms-for-EM-main/`
4. **Listo**, tenés todo igual que antes.

---

## ✅ Checklist de recuperación completa

- [ ] Clonar repo desde GitHub
- [ ] Descargar `outputs/` de Drive → poner en `arena-capstone/outputs/`
- [ ] Descargar `results_ale/` de Drive → renombrar a `results/` → poner en `arena-capstone/results/`
- [ ] Descargar `persona_adapters/` de Drive → poner en `arena-capstone/model-organisms-for-EM-main/model-organisms-for-EM-main/persona_adapters/`
- [ ] Verificar con `ls` que todo está en su lugar
- [ ] (Opcional) Configurar rclone para futuros backups

---

**¿Dudas?** Todas las carpetas mencionadas están en tu Google Drive bajo `ARENA_Capstone_models/`. El código está en GitHub en la rama `ale/dev`.
