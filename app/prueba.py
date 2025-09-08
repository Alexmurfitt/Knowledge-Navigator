import os
from dotenv import load_dotenv
# Importamos las dos funciones que necesitamos de drive_utils
from drive_untils import get_drive_service, upload_file_to_drive

def run_test():
    """
    Script de prueba que carga las variables de entorno y las pasa
    como argumentos a las funciones de utilidad de Drive.
    """
    print("--- Iniciando prueba de subida a Google Drive ---")
    
    load_dotenv()

    print("\n--- Verificando variables de entorno de Drive ---")
    cred_path = os.getenv('GOOGLE_DRIVE_CREDENTIALS_PATH')
    folder_id = os.getenv('GOOGLE_DRIVE_FOLDER_ID')
    # Lee el email del propietario desde el .env aquí
    owner_email = os.getenv('GOOGLE_ACCOUNT_EMAIL')
    
    print(f"Ruta de credenciales: {cred_path}")
    print(f"ID de la carpeta: {folder_id}")
    print(f"Email del propietario: {owner_email}")
    print("---------------------------------------------\n")

    if not all([cred_path, folder_id, owner_email]):
        print("🔴 ERROR CRÍTICO: Falta una o más variables de entorno (PATH, FOLDER_ID, EMAIL).")
        print("   Por favor, revisa tu archivo .env.")
        return

    print("Autenticando con Google Drive...")
    drive_service = get_drive_service(credentials_path=cred_path)

    if not drive_service:
        print("🔴 LA PRUEBA FALLÓ: No se pudo autenticar el servicio de Drive.")
        return
    else:
        print("✅ Servicio de Google Drive autenticado correctamente.")

    dummy_file_content = "Este es un archivo de prueba generado por el script de diagnóstico.".encode('utf-8')
    dummy_filename = "archivo_de_prueba.txt"

    print(f"\n☁️ Subiendo '{dummy_filename}' a Google Drive...")
    file_id = upload_file_to_drive(
        service=drive_service, 
        folder_id=folder_id, 
        file_content=dummy_file_content, 
        filename=dummy_filename,
        owner_email=owner_email  # Pasa el email como argumento aquí
    )

    if file_id:
        print(f"✅ Archivo '{dummy_filename}' subido con éxito a Drive. ID: {file_id}")
        print("\n" + "="*50)
        print("🎉 ¡LA PRUEBA FUE EXITOSA!")
        print("="*50)
    else:
        print("\n" + "="*50)
        print("🔴 LA PRUEBA FALLÓ.")
        print("="*50)

if __name__ == "__main__":
    run_test()








metadata = {
    # ... (tus otros metadatos)
    # NUEVO: Añadimos la fecha de subida a los metadatos de cada chunk
    "upload_date": upload_date_str 
}
    * Finalmente, en el endpoint `/upload`, al llamar a esta función, ahora le pasamos la fecha que capturamos antes:
```python
# CAMBIO: Pasamos la fecha de subida al crear los chunks
file_chunks = create_langchain_chunks(structured_blocks, filename, upload_date, bookmark_map)
