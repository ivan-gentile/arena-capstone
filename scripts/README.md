# Scripts

## Subir outputs a Google Drive

```bash
./scripts/upload_outputs_to_drive.sh
```

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

Después de eso, `./scripts/upload_outputs_to_drive.sh` sube todo `outputs/` a `arena-capstone/outputs/` en tu Drive.

- `--sync` para que Drive quede igual que tu carpeta local (borra en Drive lo que ya no está local).
