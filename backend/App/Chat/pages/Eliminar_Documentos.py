# pages/Eliminar_Documentos.py
import streamlit as st
# Importamos las funciones de nuestro módulo de lógica
from Accion_Qdrant import *

# --- Configuración de la página ---
st.set_page_config(page_title="Eliminar Documentos", page_icon="🗑️")
st.title("Gestionar Documentos de la Base de Conocimiento")

st.markdown("Aquí puedes ver los documentos cargados en la colección de Qdrant y eliminar aquellos que ya no necesites.")

collection_name = os.getenv("COLLECTION-NAME") # Define el nombre de tu colección aquí

# --- Obtener y mostrar la lista de documentos ---
try:
    # 1. Llamamos a la función de lógica para obtener los PDFs
    list_pdf = mostrar_documentos_unicos(collection_name=collection_name)

    if not list_pdf:
        st.warning("No se encontraron documentos en la colección o no se pudo acceder a Qdrant.")
    else:
        st.write(f"**Se encontraron {len(list_pdf)} documentos:**")
        
        # Usamos un formulario para agrupar el selector y el botón
        with st.form("delete_form"):
            selected_pdf = st.selectbox(
                "Selecciona el documento que deseas eliminar:",
                options=list_pdf
            )
            
            submitted = st.form_submit_button("Eliminar Documento Seleccionado", type="primary")

            if submitted:
                if selected_pdf:
                    with st.spinner(f"Eliminando '{selected_pdf}'..."):
                        # 2. Llamamos a la función de lógica para borrar el PDF
                        success = eliminar_pdf_qdrant(collection_name=collection_name,pdf_nombre= selected_pdf)
                    
                    if success:
                        st.success(f"¡El documento '{selected_pdf}' ha sido eliminado exitosamente!")
                        st.info("Recargando la lista de documentos...")
                        # Forzar un rerun para actualizar la lista de documentos mostrada
                        st.rerun()
                    else:
                        st.error(f"Ocurrió un error al intentar eliminar '{selected_pdf}'. Revisa la consola para más detalles.")
                else:
                    st.error("Por favor, selecciona un documento para eliminar.")

except Exception as e:
    st.error(f"Ocurrió un error general en la aplicación: {e}")