# Backup completo: GitHub + Google Drive

## Qué está en cada sitio

- **GitHub** (rama `ale/dev-backup`): todo el código, configs, scripts y archivos pequeños. Un solo commit, sin historial pesado, para que el push funcione.
- **Google Drive**: conviene respaldar ahí las carpetas grandes que no van a GitHub.

## Carpetas para subir a Google Drive

En tu máquina local, estas carpetas no se suben a GitHub (por tamaño). Para tener **todo** respaldado, copialas o comprimilas y subilas a GDrive:

- **`outputs/`** — checkpoints, adapters, datasets de entrenamiento (~20 GB)
- **`results/`** — resultados de evaluación, curvas, comparaciones (~6 GB)

Opcional: si querés conservar también el historial de Git en GDrive, podés comprimir la carpeta **`.git`** y subirla (solo como respaldo extra).

## Cómo usar la rama en GitHub

- La rama **`ale/dev-backup`** ya está en GitHub con todo el código.
- Tu rama **`ale/dev`** local sigue igual: con historial completo y las carpetas `outputs/` y `results/` en disco.
- Si querés que en GitHub la rama principal sea esta versión “limpia”:
  1. En el repo: **Settings → Default branch** → cambiar a `ale/dev-backup`, o
  2. Renombrar en GitHub: `ale/dev-backup` → `ale/dev` (y borrar la `ale/dev` vieja si ya no la necesitás).

## Volver a trabajar después

- Para seguir desarrollando en tu máquina: quedate en **`ale/dev`** como hasta ahora.
- Si en algún momento querés alinear GitHub con tu trabajo actual (solo código, sin outputs/results), podés repetir el proceso: crear una rama huérfana, commit con `.gitignore` que excluya `outputs/` y `results/`, y push de esa rama.
