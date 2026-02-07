# Backup completo: GitHub + Google Drive

## Qué está en cada sitio

- **GitHub** (rama `ale/dev-backup`): todo el código, configs, scripts y archivos pequeños. Un solo commit, sin historial pesado, para que el push funcione.
- **Google Drive**: `outputs/` y `results/` — usá el script que ya tenés:

  ```bash
  ./scripts/upload_outputs_to_drive.sh          # sube solo nuevos o modificados
  ./scripts/upload_outputs_to_drive.sh --sync   # sincroniza (Drive = local)
  ```

  Requiere `rclone` configurado (ver `scripts/README.md`).

## Cómo usar la rama en GitHub

- La rama **`ale/dev-backup`** ya está en GitHub con todo el código.
- Tu rama **`ale/dev`** local sigue igual: con historial completo y las carpetas `outputs/` y `results/` en disco.
- Si querés que en GitHub la rama principal sea esta versión “limpia”:
  1. En el repo: **Settings → Default branch** → cambiar a `ale/dev-backup`, o
  2. Renombrar en GitHub: `ale/dev-backup` → `ale/dev` (y borrar la `ale/dev` vieja si ya no la necesitás).

## Volver a trabajar después

- Para seguir desarrollando en tu máquina: quedate en **`ale/dev`** como hasta ahora.
- Si en algún momento querés alinear GitHub con tu trabajo actual (solo código, sin outputs/results), podés repetir el proceso: crear una rama huérfana, commit con `.gitignore` que excluya `outputs/` y `results/`, y push de esa rama.
