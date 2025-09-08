import React, { useEffect, useState, useRef } from 'react';
import { AnimatePresence, motion } from 'framer-motion';
import { FaTrash, FaEye, FaFilePdf, FaUpload, FaCalendarAlt } from 'react-icons/fa';

// NUEVO: Definimos la estructura de datos que esperamos del backend
interface DocumentData {
  name: string;
  upload_date: string;
}

// CAMBIO: El componente ahora recibe el objeto 'document' completo
interface DocumentCardProps {
  document: DocumentData;
  onDelete: (documentName: string) => void;
  isDeleting: boolean; 
}

const DocumentCard: React.FC<DocumentCardProps> = ({ document, onDelete, isDeleting }) => {
  const [hovered, setHovered] = useState(false);
  const gradientColors = ['#4A90E2', '#50E3C2', '#A0D9F7', '#F7DC6F', '#E74C3C', '#9B59B6'];

  const handleViewClick = () => {
    alert(`Functionality to view "${document.name}" will be added here.`);
  };

  return (
    <motion.div
      className="relative p-0.5 rounded-xl overflow-hidden shadow-lg"
      onHoverStart={() => setHovered(true)}
      onHoverEnd={() => setHovered(false)}
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -20 }}
      whileHover={{ scale: 1.05 }}
      transition={{ duration: 0.2, ease: "easeInOut" }}
    >
      <motion.div
        className="absolute inset-0 rounded-xl z-0"
        style={{ background: `linear-gradient(45deg, ${gradientColors.join(',')})`, backgroundSize: '400% 400%' }}
        animate={{ backgroundPosition: hovered ? ['0% 0%', '100% 0%', '100% 100%', '0% 100%', '0% 0%'] : '0% 0%' }}
        transition={{ duration: 8, ease: "linear", repeat: hovered ? Infinity : 0 }}
      />
      
      {/* CAMBIO: Reestructurado para fijar la posición de la fecha en la parte inferior */}
      <div className="relative z-10 bg-white rounded-xl p-4 flex flex-col h-full border border-gray-200 min-h-[200px]">
        {/* Contenido superior que crece para empujar el resto hacia abajo */}
        <div className="flex-grow">
          <div className="flex items-start gap-2 mb-2">
            <FaFilePdf className="text-blue-500 text-2xl flex-shrink-0 mt-1" />
            <p className="text-gray-800 font-semibold text-left break-all">
              {document.name.replace('pdf_/', '').replace('.pdf', '')}
            </p>
          </div>
        </div>

        {/* Contenido inferior con posición fija */}
        <div>
          {/* Fecha de subida */}
          <div className="flex items-center gap-2 mb-2 text-xs text-gray-500">
            <FaCalendarAlt />
            <span>Uploaded the day: {document.upload_date}</span>
          </div>

          {/* Botones de acción */}
          <div className="flex justify-end items-center gap-3 pt-2 border-t border-gray-200">
            <button onClick={handleViewClick} className="text-blue-600 hover:text-blue-800 p-2 rounded-full" title="View Document">
              <FaEye size={20} />
            </button>
            <button
              onClick={() => onDelete(document.name)}
              disabled={isDeleting}
              className="text-red-600 hover:text-red-800 p-2 rounded-full disabled:opacity-50"
              title="Delete Document"
            >
              {isDeleting ? <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-red-600"></div> : <FaTrash size={18} />}
            </button>
          </div>
        </div>
      </div>
    </motion.div>
  );
};

// --- Componente Principal ---
const Documents: React.FC = () => {
  // CAMBIO: El estado ahora almacena un array de objetos DocumentData
  const [documents, setDocuments] = useState<DocumentData[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [deletingDoc, setDeletingDoc] = useState<string | null>(null);
  const [isUploading, setIsUploading] = useState(false);

  const fileInputRef = useRef<HTMLInputElement>(null);

  const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:5000';
  const COLLECTION_NAME = "Knowledge-Navigator"; 

  const fetchDocuments = async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await fetch(`${API_BASE_URL}/documentos_unicos/${COLLECTION_NAME}`);
      if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
      // CAMBIO: La respuesta ahora es un array de objetos
      const data: DocumentData[] = await response.json();
      setDocuments(data);
    } catch (e: any) {
      setError("Failed to load documents. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchDocuments();
  }, []);

  const handleDeleteDocument = async (pdfName: string) => {
    if (!window.confirm(`Are you sure you want to delete "${pdfName.replace('pdf_/', '').replace('.pdf', '')}"?`)) return;
    setDeletingDoc(pdfName);
    try {
      const response = await fetch(`${API_BASE_URL}/delete?collection_name=${COLLECTION_NAME}&pdf_nombre=${pdfName}`, {
        method: 'DELETE',
      });
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || `HTTP error!`);
      }
      await fetchDocuments(); 
    } catch (e: any) {
      alert(`Error deleting document: ${e.message}`);
    } finally {
      setDeletingDoc(null);
    }
  };

  const handleUploadClick = () => {
    fileInputRef.current?.click();
  };

  const handleFileChange = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const files = event.target.files;
    if (!files || files.length === 0) return;
    setIsUploading(true);
    setError(null);
    const formData = new FormData();
    for (let i = 0; i < files.length; i++) {
      formData.append("files", files[i]);
    }
    try {
      const response = await fetch(`${API_BASE_URL}/upload`, { method: 'POST', body: formData });
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'File upload failed');
      }
      await fetchDocuments();
    } catch (e: any) {
      setError(`Upload failed: ${e.message}`);
    } finally {
      setIsUploading(false);
      if (fileInputRef.current) fileInputRef.current.value = "";
    }
  };

  return (
    <div className="flex flex-col items-center min-h-screen bg-gradient-to-br from-blue-50 to-blue-200 p-8 text-gray-800">
      <input type="file" multiple accept=".pdf" ref={fileInputRef} onChange={handleFileChange} style={{ display: 'none' }} />
      
      <div className="w-full max-w-7xl flex justify-between items-center mb-10">
        <h1 className="text-5xl font-extrabold text-blue-700">Your Documents</h1>
        <button onClick={handleUploadClick} disabled={isUploading} className="bg-blue-600 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded-lg flex items-center gap-2 shadow-lg disabled:opacity-75">
          {isUploading ? (<><div className="animate-spin rounded-full h-5 w-5 border-b-2 border-white"></div>Uploading...</>) : (<><FaUpload />Upload Files</>)}
        </button>
      </div>

      {error && <div className="bg-red-100 border border-red-400 text-red-700 px-6 py-4 rounded-lg shadow-xl text-center mb-4"><p>{error}</p></div>}
      
      {loading && <div className="text-2xl text-blue-600">Loading documents...</div>}

      {!loading && !error && documents.length === 0 && (
        <div className="text-2xl text-blue-500 mt-10 p-6 border border-dashed border-blue-300 rounded-lg text-center">
          <p>No documents found yet.</p>
        </div>
      )}

      <AnimatePresence>
        {!loading && documents.length > 0 && (
          <motion.div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-8 w-full max-w-7xl mt-8" initial="hidden" animate="visible" variants={{ visible: { transition: { staggerChildren: 0.07 } } }}>
            {/* CAMBIO: Mapeamos el nuevo array de objetos */}
            {documents.map((doc) => (
              <DocumentCard key={doc.name} document={doc} onDelete={handleDeleteDocument} isDeleting={deletingDoc === doc.name} />
            ))}
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
};

export default Documents;

