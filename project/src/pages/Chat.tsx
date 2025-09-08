import React, { useState, useRef, useEffect, useCallback } from 'react';
import { Send, Trash2, Globe, Info, X } from 'lucide-react';
import botAvatar from '../images/asistant.png';

// Importaciones para el fondo animado
import Particles from "react-tsparticles";
import { loadSlim } from "tsparticles-slim";
import type { Engine } from "tsparticles-engine";
import { options } from "./particlesConfig"; // Asumimos que el archivo de config está en la misma carpeta

// --- Tipos y Datos ---
interface Source {
    page_content: string;
    metadata: {
        document_name_id: string;
        page_number: number;
        content_type: string;
        title_hierarchy: Record<string, string>;
    };
}

interface Message {
  id: string;
  type: 'user' | 'assistant';
  content: string;
  timestamp: Date;
  hasAdditionalInfo?: boolean;
  additionalInfo?: {
    fullText?: string;
    sources?: Source[];
  };
}

// --- Componente de Estilos (sin cambios) ---
const CustomStyles = () => (
  <style>{`
    .custom-scrollbar::-webkit-scrollbar { width: 8px; }
    .custom-scrollbar::-webkit-scrollbar-track { background: transparent; }
    .custom-scrollbar::-webkit-scrollbar-thumb { background: #38bdf8; border-radius: 10px; }
    .custom-scrollbar::-webkit-scrollbar-thumb:hover { background: #0ea5e9; }
    @keyframes futuristic-wave-animation {
      0%, 100% { transform: scaleY(0.2); background-color: #38bdf8; box-shadow: 0 0 5px #38bdf8, 0 0 15px #38bdf8; }
      50% { transform: scaleY(1); background-color: #67e8f9; box-shadow: 0 0 10px #67e8f9, 0 0 30px #67e8f9; }
    }
    .wave-line {
      width: 4px; height: 40px; margin: 0 2px; border-radius: 2px;
      animation-name: futuristic-wave-animation;
      animation-timing-function: ease-in-out;
      animation-iteration-count: infinite;
    }
 `}</style>
);

// --- Componente de Animación (sin cambios) ---
const FuturisticWaveLoader = () => (
    <div className="flex items-center justify-center h-full">
      {[...Array(7)].map((_, i) => (
        <div key={i} className="wave-line" style={{ animationDuration: `${0.8 + i * 0.1}s`, animationDelay: `${i * 0.15}s` }} />
      ))}
    </div>
);

// --- Componentes Secundarios (sin cambios) ---
const AssistantAvatar = () => (
    <div className="w-8 h-8 rounded-full flex items-center justify-center flex-shrink-0 overflow-hidden">
        <img src={botAvatar} alt="Assistant Avatar" className="w-full h-full object-cover" />
    </div>
);

const ChatMessage: React.FC<{ message: Message; onShowInfo: () => void }> = ({ message, onShowInfo }) => {
    const isAssistant = message.type === 'assistant';
    const bubbleClasses = isAssistant ? 'bg-gray-100 text-gray-800 border border-sky-300' : 'bg-blue-600 text-white';
    const alignmentClasses = isAssistant ? 'items-start' : 'items-end';
  
    return (
      <div className={`flex flex-col ${alignmentClasses} group`}>
        <div className="flex items-start space-x-3 max-w-2xl">
          {isAssistant && <AssistantAvatar />}
          <div className={`relative px-4 py-3 rounded-2xl ${bubbleClasses} ${isAssistant ? 'rounded-tl-none' : 'rounded-br-none'} shadow-sm`}>
            <p className="text-sm whitespace-pre-wrap">{message.content}</p>
          </div>
          {isAssistant && message.hasAdditionalInfo && (
            <button onClick={onShowInfo} className="p-2 text-sky-500 hover:text-sky-600 opacity-0 group-hover:opacity-100 transition-opacity self-center">
              <Info className="w-4 h-4" />
            </button>
          )}
        </div>
         <p className="text-xs text-gray-400 mt-1 px-2">{message.timestamp.toLocaleTimeString()}</p>
      </div>
    );
};
  
const InfoPanel: React.FC<{ isOpen: boolean; onClose: () => void; info: any }> = ({ isOpen, onClose, info }) => {
    const getProcessedSources = () => {
      if (!info?.sources) return [];
      const uniqueSources = info.sources.reduce((acc: Source[], current: Source) => {
        const isDuplicate = acc.some(item => 
          item.metadata.document_name_id === current.metadata.document_name_id &&
          item.metadata.page_number === current.metadata.page_number
        );
        if (!isDuplicate) acc.push(current);
        return acc;
      }, []);
      return uniqueSources.slice(0, 3);
    };
    const processedSources = getProcessedSources();
  
    return (
      <aside className={`transform top-0 left-0 h-full bg-white border-r border-gray-200 transition-all duration-300 ease-in-out flex flex-col ${isOpen ? 'w-96 p-6' : 'w-0 p-0' } overflow-hidden`}>
          <div className="flex items-center justify-between mb-6">
              <h3 className="text-lg font-semibold text-gray-900">Additional Information</h3>
              <button onClick={onClose} className="text-gray-400 hover:text-gray-700">
                  <X className="w-5 h-5" />
              </button>
          </div>
          <div className="flex-1 overflow-y-auto text-sm text-gray-600 space-y-4 custom-scrollbar">
              {info ? (
                  <>
                      {info.fullText && <p className="leading-relaxed">{info.fullText}</p>}
                      {processedSources.length > 0 && (
                           <div className="space-y-3 pt-4 mt-4 border-t border-gray-200">
                              <h4 className="font-semibold text-gray-800 mb-2">Fonts:</h4>
                              <ul className="space-y-2">
                                  {processedSources.map((source: Source, index: number) => (
                                      <li key={index} className="p-2 border rounded-md bg-gray-50 text-xs">
                                          <p className="font-medium text-blue-700 truncate">{source.metadata.document_name_id} (Pág. {source.metadata.page_number})</p>
                                      </li>
                                  ))}
                              </ul>
                          </div>
                      )}
                  </>
              ) : <p>No hay información para mostrar.</p>}
          </div>
      </aside>
    );
};

// --- Componente Principal del Chat ---
const App: React.FC = () => {
  const [messages, setMessages] = useState<Message[]>([
    {
      id: '1',
      type: 'assistant',
      content: '¡Hola! Soy tu Asistente de Documentos. Puedo ayudarte a encontrar información en tus archivos. ¿Qué te gustaría saber?',
      timestamp: new Date()
    }
  ]);
  const [inputMessage, setInputMessage] = useState('');
  const [internetEnabled, setInternetEnabled] = useState(false);
  const [isTyping, setIsTyping] = useState(false);
  const [isInfoPanelOpen, setIsInfoPanelOpen] = useState(false);
  const [selectedMessageInfo, setSelectedMessageInfo] = useState<any>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  
  const API_BASE_URL = "http://localhost:5000";

  // Función necesaria para inicializar el motor de partículas
  const particlesInit = useCallback(async (engine: Engine) => {
    await loadSlim(engine);
  }, []);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const handleSendMessage = async () => {
    if (!inputMessage.trim()) return;
    const userMessage: Message = { id: Date.now().toString(), type: 'user', content: inputMessage, timestamp: new Date() };
    setMessages(prev => [...prev, userMessage]);
    setInputMessage('');
    setIsTyping(true);
    
    try {
      const response = await fetch(`${API_BASE_URL}/ask`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ question: inputMessage, use_internet: internetEnabled }),
      });
      if (!response.ok) throw new Error(`Error de la API: ${response.statusText}`);
      const data = await response.json();
      const assistantMessage: Message = {
        id: (Date.now() + 1).toString(), type: 'assistant', content: data.answer, timestamp: new Date(),
        hasAdditionalInfo: data.additional_info || (data.sources && data.sources.length > 0),
        additionalInfo: { fullText: data.additional_info, sources: data.sources }
      };
      setMessages(prev => [...prev, assistantMessage]);
    } catch (error) {
        console.error("Error al contactar con la API:", error);
        const errorMessage: Message = {
            id: (Date.now() + 1).toString(), type: 'assistant',
            content: 'Lo siento, ha ocurrido un error al procesar tu solicitud. Por favor, inténtalo de nuevo más tarde.',
            timestamp: new Date(),
        };
        setMessages(prev => [...prev, errorMessage]);
    } finally {
        setIsTyping(false);
    }
  };

  const handleClearChat = () => {
    setIsInfoPanelOpen(false);
    setMessages([{ id: '1', type: 'assistant', content: 'Historial de chat borrado. ¿Cómo puedo ayudarte hoy?', timestamp: new Date() }]);
  };

  const handleShowInfo = (message: Message) => {
    setSelectedMessageInfo(message.additionalInfo);
    setIsInfoPanelOpen(true);
  };
  
  const handleCloseInfoPanel = () => {
      setIsInfoPanelOpen(false);
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  return (
    <div className="h-[calc(100vh-4rem)] w-full bg-transparent flex overflow-hidden font-sans relative">
      <Particles
        id="tsparticles"
        init={particlesInit}
        options={options}
        className="fixed top-0 left-0 w-full h-full z-0"
      />
      
      <div className="relative z-10 flex flex-1">
        <CustomStyles />
        <InfoPanel
          isOpen={isInfoPanelOpen}
          onClose={handleCloseInfoPanel}
          info={selectedMessageInfo}
        />
        
        <main className={`flex-1 flex items-center p-4 transition-all duration-300 ease-in-out ${isInfoPanelOpen ? 'justify-start' : 'justify-center'}`}>
          <div className="w-full max-w-4xl h-full flex flex-col">
              {/* CAMBIO: Contenedor del borde degradado RESTAURADO */}
              <div className="flex-1 flex flex-col min-h-0 bg-gradient-to-br from-blue-400 via-cyan-400 to-indigo-500 rounded-2xl p-1 shadow-2xl">
                {/* CAMBIO: Contenedor interior AHORA es translúcido */}
                <div className="w-full h-full flex flex-col bg-white/80 backdrop-blur-sm rounded-xl overflow-hidden">
                  {/* Cabecera del Chat */}
                  <div className="bg-transparent border-b border-white/30 p-4">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-3">
                        <img src={botAvatar} alt="Assistant Avatar" className="w-20 h-16" />
                        <h1 className="text-xl font-semibold text-gray-900">DOCUMENT ASSISTANT</h1>
                      </div>
                      <div className="flex items-center space-x-4">
                        <div className="flex items-center space-x-2">
                          <Globe className={`w-4 h-4 ${internetEnabled ? 'text-blue-600' : 'text-gray-400'}`} />
                          <span className="text-sm text-gray-600">Internet</span>
                          <button onClick={() => setInternetEnabled(!internetEnabled)} className={`relative inline-flex h-5 w-9 items-center rounded-full transition-colors duration-200 ${internetEnabled ? 'bg-blue-600' : 'bg-gray-300'}`}>
                            <span className={`inline-block h-3 w-3 transform rounded-full bg-white transition-transform duration-200 ${internetEnabled ? 'translate-x-5' : 'translate-x-1'}`} />
                          </button>
                        </div>
                      </div>
                    </div>
                  </div>

                  {/* Área de Mensajes (con scroll) */}
                  <div className="flex-1 min-h-0 overflow-y-auto bg-transparent custom-scrollbar">
                    <div className="p-4 space-y-4">
                      {messages.map((message) => (
                        <ChatMessage key={message.id} message={message} onShowInfo={() => handleShowInfo(message)} />
                      ))}
                      {isTyping && (
                        <div className="flex items-start space-x-3">
                          <AssistantAvatar />
                          <div className="bg-white px-4 py-3 rounded-xl rounded-tl-none shadow-sm flex items-center justify-center overflow-hidden" style={{ minWidth: '150px' }}>
                            <FuturisticWaveLoader />
                          </div>
                        </div>
                      )}
                      <div ref={messagesEndRef} />
                    </div>
                  </div>

                  {/* Área de Entrada de Texto */}
                  <div className="bg-transparent border-t border-white/30 p-4">
                    <div className="flex items-end space-x-3">
                      <div className="flex-1">
                        <textarea value={inputMessage} onChange={(e) => setInputMessage(e.target.value)} onKeyPress={handleKeyPress} placeholder="Ask anything about your documents..." className="w-full px-4 py-3 border border-gray-300 rounded-xl resize-none focus:outline-none focus:ring-2 focus:ring-blue-500 h-20"/>
                      </div>
                      <div className="flex flex-col space-y-2">
                        <button onClick={handleClearChat} className="p-3 bg-red-600 hover:bg-red-700 text-white rounded-xl transition-all hover:shadow-lg transform hover:scale-105" title="Limpiar historial">
                          <Trash2 className="w-5 h-5" />
                        </button>
                        <button onClick={handleSendMessage} disabled={!inputMessage.trim()} className="p-3 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-300 text-white rounded-xl transition-all hover:shadow-lg disabled:cursor-not-allowed transform hover:scale-105">
                          <Send className="w-5 h-5" />
                        </button>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
          </div>
        </main>
      </div>
    </div>
  );
};

export default App;

