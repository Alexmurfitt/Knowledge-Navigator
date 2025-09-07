import React, { useEffect, useState, useRef } from 'react';
import { AnimatePresence, motion } from 'framer-motion';
import { FaTrash, FaEye, FaFilePdf, FaUpload } from 'react-icons/fa';

interface DocumentCardProps {
  documentName: string;
  onDelete: (documentName: string) => void;
  isDeleting: boolean; 
}

const DocumentCard: React.FC<DocumentCardProps> = ({ documentName, onDelete, isDeleting }) => {
  const [hovered, setHovered] = useState(false);
  const gradientColors = ['#4A90E2', '#50E3C2', '#A0D9F7', '#F7DC6F', '#E74C3C', '#9B59B6'];

  const handleViewClick = () => {
    alert(`Functionality to view "${documentName}" will be added here.`);
  };

  return (
    // CAMBIO AQUÍ 1: Se ha añadido la propiedad "whileHover" para el efecto de zoom
    <motion.div
      className="relative p-0.5 rounded-xl overflow-hidden shadow-lg"
      onHoverStart={() => setHovered(true)}
      onHoverEnd={() => setHovered(false)}
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -20 }}
      whileHover={{ scale: 1.05 }} // <-- Aumenta el tamaño de la tarjeta al pasar el ratón
      transition={{ duration: 0.2, ease: "easeInOut" }} // <-- Transición suave para todos los efectos
    >
      <motion.div
        className="absolute inset-0 rounded-xl z-0"
        style={{
          background: `linear-gradient(45deg, ${gradientColors.join(',')})`,
          backgroundSize: '400% 400%',
        }}
        animate={{
          backgroundPosition: hovered 
            ? ['0% 0%', '100% 0%', '100% 100%', '0% 100%', '0% 0%']
            : '0% 0%',
        }}
        transition={{
          duration: hovered ? 8 : 0.5,
          ease: "linear",
          repeat: hovered ? Infinity : 0,
        }}
      />
      
      <div className="relative z-10 bg-white rounded-xl p-4 flex flex-col justify-between h-full border border-gray-200 min-h-[180px]">
        <div className="absolute top-2 right-2 bg-green-100 text-green-800 text-xs font-semibold px-2.5 py-0.5 rounded-full">
          Uploaded
        </div>
        <div className="flex-grow flex flex-col justify-center">
          <div className="flex items-start gap-2 mb-2"> {/* Cambiado a items-start para mejor alineación si hay wrap */}
            <FaFilePdf className="text-blue-500 text-2xl flex-shrink-0 mt-1" />
            {/* CAMBIO AQUÍ 2: Se ha cambiado "break-words" por "break-all" para asegurar el salto de línea */}
            <p className="text-gray-800 font-semibold text-left break-all">
              {documentName.replace('pdf_/', '').replace('.pdf', '')}
            </p>
          </div>
        </div>
        <div className="flex justify-end items-center gap-3 mt-4 pt-2 border-t border-gray-200">
          <button
            onClick={handleViewClick}
            className="text-blue-600 hover:text-blue-800 p-2 rounded-full transition-colors duration-200"
            title="View Document"
          >
            <FaEye size={20} />
          </button>
          <button
            onClick={() => onDelete(documentName)}
            disabled={isDeleting}
            className="text-red-600 hover:text-red-800 p-2 rounded-full transition-colors duration-200 disabled:opacity-50 disabled:cursor-not-allowed"
            title="Delete Document"
          >
            {isDeleting ? <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-red-600"></div> : <FaTrash size={18} />}
          </button>
        </div>
      </div>
    </motion.div>
  );
};

// --- El resto del componente Documents permanece igual ---
const Documents: React.FC = () => {
  const [documents, setDocuments] = useState<string[]>([]);
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
      const data: string[] = await response.json();
      setDocuments(data);
    } catch (e: any) {
      console.error("Failed to fetch documents:", e);
      setError("Failed to load documents. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchDocuments();
  }, []);

  const handleDeleteDocument = async (pdfName: string) => {
    if (!window.confirm(`Are you sure you want to delete "${pdfName.replace('pdf_/', '').replace('.pdf', '')}"? This action cannot be undone.`)) {
      return;
    }
    setDeletingDoc(pdfName);
    try {
      const response = await fetch(`${API_BASE_URL}/delete?collection_name=${COLLECTION_NAME}&pdf_nombre=${pdfName}`, {
        method: 'DELETE',
      });
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || `HTTP error! status: ${response.status}`);
      }
      await fetchDocuments(); 
    } catch (e: any) {
      console.error("Failed to delete document:", e);
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
      const response = await fetch(`${API_BASE_URL}/upload`, {
        method: 'POST',
        body: formData,
      });
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'File upload failed');
      }
      alert('Files uploaded successfully!');
      await fetchDocuments();
    } catch (e: any) {
      console.error('Upload error:', e);
      setError(`Upload failed: ${e.message}`);
    } finally {
      setIsUploading(false);
      if (fileInputRef.current) {
        fileInputRef.current.value = "";
      }
    }
  };

  return (
    <div className="flex flex-col items-center min-h-screen bg-gradient-to-br from-blue-50 to-blue-200 p-8 text-gray-800">
      <input type="file" multiple accept=".pdf" ref={fileInputRef} onChange={handleFileChange} style={{ display: 'none' }} />
      
      <div className="w-full max-w-7xl flex justify-between items-center mb-10">
        <h1 className="text-5xl font-extrabold text-blue-700">Your Documents</h1>
        <button
          onClick={handleUploadClick}
          disabled={isUploading}
          className="bg-blue-600 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded-lg flex items-center gap-2 shadow-lg transition-transform transform hover:scale-105 disabled:opacity-75 disabled:cursor-not-allowed"
        >
          {isUploading ? (
            <><div className="animate-spin rounded-full h-5 w-5 border-b-2 border-white"></div>Uploading...</>
          ) : (
            <><FaUpload />Upload Files</>
          )}
        </button>
      </div>

      {error && <div className="bg-red-100 border border-red-400 text-red-700 px-6 py-4 rounded-lg shadow-xl text-center text-xl max-w-lg mb-4"><p className="font-bold mb-2">Error!</p><p>{error}</p></div>}
      {(loading && !isUploading) && <div className="flex items-center space-x-3 text-2xl text-blue-600"><svg className="animate-spin h-8 w-8 text-blue-500" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24"><circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle><path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path></svg><span>Loading documents...</span></div>}
      {!loading && !error && documents.length === 0 && <div className="text-2xl text-blue-500 mt-10 p-6 border border-dashed border-blue-300 rounded-lg max-w-xl text-center"><p>No documents found yet.</p><p className="mt-2 text-lg">Upload some documents to start chatting!</p></div>}

      <AnimatePresence>
        {!loading && documents.length > 0 && (
          <motion.div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-8 w-full max-w-7xl mt-8" initial="hidden" animate="visible" variants={{ visible: { transition: { staggerChildren: 0.07 } } }}>
            {documents.map((docName) => (
              <DocumentCard key={docName} documentName={docName} onDelete={handleDeleteDocument} isDeleting={deletingDoc === docName} />
            ))}
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
};

export default Documents;