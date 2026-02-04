#!/usr/bin/env bash
#
# Sube outputs/ y results/ a Google Drive.
# - outputs/ → ARENA_Capstone_models/outputs
# - results/ → ARENA_Capstone_models/results_ale (subcarpeta separada para no pisar results existente)
# Primera vez: instalar rclone y vincular tu cuenta (ver abajo).
#
# Uso:
#   ./scripts/upload_outputs_to_drive.sh          # sube outputs/ y results/ (solo nuevos o modificados)
#   ./scripts/upload_outputs_to_drive.sh --sync   # sincroniza (borra en Drive lo que ya no está local)
#

set -e

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUTPUTS_DIR="$REPO_ROOT/outputs"
RESULTS_DIR="$REPO_ROOT/results"
REMOTE_NAME="${RCLONE_REMOTE:-gdrive}"
DRIVE_OUTPUTS="ARENA_Capstone_models/outputs"
DRIVE_RESULTS="ARENA_Capstone_models/results_ale"

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

# --- Crear carpetas en Drive si no existen (evita error 404)
echo "Creando carpetas en Drive si no existen..."
if ! rclone mkdir "${REMOTE_NAME}:ARENA_Capstone_models" 2>/dev/null; then
  echo "  (carpeta ARENA_Capstone_models ya existe o se creó)"
fi
if ! rclone mkdir "${REMOTE_NAME}:${DRIVE_OUTPUTS}" 2>/dev/null; then
  echo "  (carpeta outputs ya existe o se creó)"
fi
if ! rclone mkdir "${REMOTE_NAME}:${DRIVE_RESULTS}" 2>/dev/null; then
  echo "  (carpeta results_ale ya existe o se creó)"
fi

# --- Confirmación de seguridad para --sync
if [[ "$1" == "--sync" ]]; then
  echo ""
  echo "=========================================="
  echo "⚠️  ADVERTENCIA: Modo --sync activado"
  echo "=========================================="
  echo ""
  echo "El modo --sync borrará en Google Drive cualquier archivo"
  echo "que NO exista localmente en outputs/ y results/."
  echo ""
  echo "Si borraste archivos localmente pero querés mantenerlos"
  echo "en Drive, NO uses --sync (ejecutá sin argumentos)."
  echo ""
  echo "Archivos actuales en local:"
  echo "  outputs/: $(du -sh "$OUTPUTS_DIR" 2>/dev/null | cut -f1)"
  echo "  results/: $(du -sh "$RESULTS_DIR" 2>/dev/null | cut -f1)"
  echo ""
  read -p "¿Continuar con --sync? (escribí 'SI' para confirmar): " confirmacion
  echo ""
  
  if [[ "$confirmacion" != "SI" ]]; then
    echo "Operación cancelada. No se realizaron cambios."
    exit 0
  fi
  
  echo "Confirmado. Procediendo con --sync..."
  echo ""
fi

# --- Subir outputs/
if [[ "$1" == "--sync" ]]; then
  echo "Sincronizando outputs/ → ${REMOTE_NAME}:${DRIVE_OUTPUTS}/ (los cambios en Drive se reflejarán)."
  rclone sync "$OUTPUTS_DIR" "${REMOTE_NAME}:${DRIVE_OUTPUTS}" --progress -v
else
  echo "Subiendo outputs/ → ${REMOTE_NAME}:${DRIVE_OUTPUTS}/ (solo archivos nuevos o modificados)."
  if ! rclone copy "$OUTPUTS_DIR" "${REMOTE_NAME}:${DRIVE_OUTPUTS}" --progress -v; then
    echo ""
    echo "Si el error es 404 (File not found), probablemente root_folder_id está mal en rclone."
    echo "Corregilo así:  rclone config  →  e (edit)  →  gdrive  →  root_folder_id: dejalo VACÍO (Enter)."
    echo "Luego volvé a ejecutar este script."
    exit 1
  fi
fi

# --- Subir results/ → results_ale
if [[ "$1" == "--sync" ]]; then
  echo "Sincronizando results/ → ${REMOTE_NAME}:${DRIVE_RESULTS}/"
  rclone sync "$RESULTS_DIR" "${REMOTE_NAME}:${DRIVE_RESULTS}" --progress -v
else
  echo "Subiendo results/ → ${REMOTE_NAME}:${DRIVE_RESULTS}/"
  rclone copy "$RESULTS_DIR" "${REMOTE_NAME}:${DRIVE_RESULTS}" --progress -v
fi

echo ""
echo "Listo. Revisá en Drive: ${DRIVE_OUTPUTS}/ y ${DRIVE_RESULTS}/"
