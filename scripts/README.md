# Scripts

## Subir outputs y results a Google Drive

```bash
./scripts/upload_outputs_to_drive.sh
```

Sube:
- **outputs/** → `ARENA_Capstone_models/outputs`
- **results/** → `ARENA_Capstone_models/results_ale` (carpeta separada para no pisar un `results` existente)

**Primera vez:** hay que vincular tu cuenta de Google (solo una vez):

```bash
rclone config
```

- `n` (new remote)  
- name: `gdrive`  
- Storage: **Google Drive**  
- client_id / secret: Enter  
- scope: **1** (Full access)  
- config: **n** (autoconfig) → se abre el navegador, iniciás sesión y autorizás  
- team drive: **n**  
- `q` (quit)

Después de eso, el script sube `outputs/` y `results/` a **My Drive / ARENA_Capstone_models /** (en `outputs` y `results_ale`).

- `--sync` para que Drive quede igual que tu carpeta local (borra en Drive lo que ya no está local).
