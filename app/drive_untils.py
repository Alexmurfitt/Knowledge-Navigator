import os
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseUpload
import io

SCOPES = ['https://www.googleapis.com/auth/drive.file']

def get_drive_service(credentials_path: str):
    """
    Autentica y devuelve el objeto de servicio de la API de Google Drive.
    """
    if not credentials_path or not os.path.exists(credentials_path):
        print(f"❌ Error: El archivo de credenciales no se encuentra en la ruta: {credentials_path}")
        return None
    try:
        creds = Credentials.from_service_account_file(credentials_path, scopes=SCOPES)
        return build('drive', 'v3', credentials=creds)
    except Exception as e:
        print(f"❌ Error al autenticar con Google Drive: {e}")
        return None

def upload_file_to_drive(service, folder_id: str, file_content: bytes, filename: str, owner_email: str):
    """
    Sube un archivo a Drive y transfiere la propiedad.
    Ahora recibe el email del propietario como un argumento.
    """
    if not service or not folder_id:
        print("❌ Error: El servicio de Drive o el ID de la carpeta no están configurados.")
        return None
        
    if not owner_email:
        print("❌ Error: No se proporcionó un email de propietario para transferir la propiedad del archivo.")
        return None

    try:
        file_metadata = {
            'name': filename,
            'parents': [folder_id]
        }
        mime_type = 'application/pdf' if filename.lower().endswith('.pdf') else 'text/plain'
        media = MediaIoBaseUpload(io.BytesIO(file_content), mimetype=mime_type, resumable=True)
        
        request = service.files().create(
            body=file_metadata,
            media_body=media,
            fields='id'
        )
        file = request.execute()
        file_id = file.get('id')

        if not file_id:
            print("❌ Error: La subida del archivo no devolvió un ID.")
            return None

        print(f"📄 Archivo subido temporalmente con ID: {file_id}. Transfiriendo propiedad...")

        permission = {
            'type': 'user',
            'role': 'owner',
            'emailAddress': owner_email  # Usa el email pasado como argumento
        }
        service.permissions().create(
            fileId=file_id,
            body=permission,
            transferOwnership=True
        ).execute()

        print(f"✅ Propiedad de '{filename}' transferida a {owner_email}.")
        return file_id

    except Exception as e:
        print(f"❌ Error durante el proceso de subida o transferencia en Drive: {e}")
        return None

