#!/usr/bin/env bash
#
# Sube la carpeta outputs/ a Google Drive (solo lo que no está en git: checkpoints, modelos, etc.).
# Primera vez: instalar rclone y vincular tu cuenta (ver abajo).
#
# Uso:
#   ./scripts/upload_outputs_to_drive.sh          # sube todo outputs/
#   ./scripts/upload_outputs_to_drive.sh --sync   # sincroniza (borra en Drive lo que ya no está local)
#

set -e

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUTPUTS_DIR="$REPO_ROOT/outputs"
REMOTE_NAME="${RCLONE_REMOTE:-gdrive}"
DRIVE_PATH="arena-capstone/outputs"

# --- Comprobar rclone
if ! command -v rclone &>/dev/null; then
  echo "rclone no está instalado."
  echo ""
  echo "Instalación rápida:"
  echo "  Linux/macOS:  curl https://rclone.org/install.sh | sudo bash"
  echo "  O descarga:   https://rclone.org/downloads/"
  echo ""
  echo "Luego configura Google Drive una sola vez:"
  echo "  rclone config"
  echo "  → n (new remote) → nombre: gdrive → tipo: drive → resto por defecto → autoconfig (abre el navegador, inicia sesión)."
  exit 1
fi

# --- Comprobar que el remote existe
if ! rclone listremotes 2>/dev/null | grep -q "^${REMOTE_NAME}:$"; then
  echo "No hay un remote llamado '${REMOTE_NAME}'."
  echo ""
  echo "Configuración (una sola vez):"
  echo "  rclone config"
  echo "  → n (new remote)"
  echo "  → name: gdrive"
  echo "  → Storage: Google Drive"
  echo "  → client_id / secret: Enter (por defecto)"
  echo "  → scope: 1 (Full access)"
  echo "  → config: n (autoconfig)"
  echo "  → Abre el navegador, inicia sesión con tu cuenta de Google, autoriza."
  echo "  → team drive: n"
  echo "  → q (quit)"
  echo ""
  echo "Si usas otro nombre de remote, ejecuta: RCLONE_REMOTE=tu_nombre $0"
  exit 1
fi

# --- Subir
if [[ "$1" == "--sync" ]]; then
  echo "Sincronizando outputs/ → ${REMOTE_NAME}:${DRIVE_PATH}/ (los cambios en Drive se reflejarán)."
  rclone sync "$OUTPUTS_DIR" "${REMOTE_NAME}:${DRIVE_PATH}" --progress -v
else
  echo "Subiendo outputs/ → ${REMOTE_NAME}:${DRIVE_PATH}/ (solo archivos nuevos o modificados)."
  rclone copy "$OUTPUTS_DIR" "${REMOTE_NAME}:${DRIVE_PATH}" --progress -v
fi

echo ""
echo "Listo. Revisá en Drive: ${DRIVE_PATH}/"
