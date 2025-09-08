import React, { useCallback } from 'react';
import { ArrowRight, Brain, Shield, Zap, FileText, MessageSquare, BarChart3 } from 'lucide-react';
import { useNavigation } from '../hooks/useNavigation';
import { useAuth } from '../hooks/useAuth';

// Importaciones para el fondo animado
import Particles from "react-tsparticles";
import { loadSlim } from "tsparticles-slim";
import type { Engine } from "tsparticles-engine";
import { options } from "./particlesConfig"; // Asumimos que el archivo de config está en la misma carpeta

const Home: React.FC = () => {
  const { setCurrentPage } = useNavigation();
  const { isAuthenticated, login } = useAuth();

  // Función necesaria para inicializar el motor de partículas
  const particlesInit = useCallback(async (engine: Engine) => {
    await loadSlim(engine);
  }, []);

  const handleStartChatting = () => {
    if (isAuthenticated) {
      setCurrentPage('chat');
    } else {
      login();
    }
  };

  const features = [
    {
      icon: Brain,
      title: 'AI-Powered Analysis',
      description: 'Advanced RAG technology extracts insights from your documents with human-like understanding.'
    },
    {
      icon: Shield,
      title: 'Enterprise Security',
      description: 'Bank-grade security ensures your sensitive documents remain private and protected.'
    },
    {
      icon: Zap,
      title: 'Instant Responses',
      description: 'Get immediate answers from your document library with lightning-fast query processing.'
    }
  ];

  return (
    <div className="min-h-screen">
      {/* CAMBIO: Componente de partículas con posición FIJA */}
      <Particles
        id="tsparticles"
        init={particlesInit}
        options={options}
        className="fixed top-0 left-0 w-full h-full z-0"
      />
      
      {/* CAMBIO: Todo el contenido de la página ahora es relativo y está por encima del fondo */}
      <div className="relative z-10">
        {/* Hero Section (SIN CAMBIOS EN EL CONTENIDO) */}
        <section className="py-20 px-4 sm:px-6 lg:px-8">
            <div className="max-w-7xl mx-auto">
                <div className="text-center">
                <h1 className="text-4xl sm:text-5xl lg:text-6xl font-bold text-gray-900 mb-6">
                    Transform Your Enterprise
                    <span className="block bg-gradient-to-r from-blue-600 to-blue-800 bg-clip-text text-transparent">
                    Documents into Intelligence
                    </span>
                </h1>
                <p className="text-xl text-gray-600 mb-8 max-w-3xl mx-auto leading-relaxed">
                    Knowledge Navigator empowers your organization with cutting-edge RAG technology, 
                    turning static PDFs into dynamic, intelligent resources that understand and respond 
                    to natural language queries.
                </p>
                
                <div className="flex flex-col sm:flex-row items-center justify-center gap-4 mb-16">
                    <button
                        onClick={handleStartChatting}
                        className="group flex items-center space-x-3 px-8 py-4 bg-blue-600 hover:bg-blue-700 text-white rounded-xl font-semibold transition-all duration-300 hover:shadow-xl hover:scale-105"
                    >
                        <MessageSquare className="w-5 h-5" />
                        <span>Start Chatting</span>
                        <ArrowRight className="w-4 h-4 group-hover:translate-x-1 transition-transform duration-200" />
                    </button>
                    <button
                        onClick={() => setCurrentPage('documents')}
                        className="flex items-center space-x-3 px-8 py-4 border-2 border-blue-600 text-blue-600 hover:bg-blue-600 hover:text-white rounded-xl font-semibold transition-all duration-300"
                    >
                        <FileText className="w-5 h-5" />
                        <span>Upload Documents</span>
                    </button>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-3 gap-8 max-w-5xl mx-auto">
                    {features.map((feature, index) => {
                        const Icon = feature.icon;
                        return (
                        <div
                            key={index}
                            className="bg-white/80 backdrop-blur-sm p-6 rounded-2xl shadow-sm border border-gray-200 hover:shadow-md transition-all duration-300 hover:-translate-y-1"
                        >
                            <div className="w-12 h-12 bg-blue-100 rounded-xl flex items-center justify-center mb-4 mx-auto">
                                <Icon className="w-6 h-6 text-blue-600" />
                            </div>
                            <h3 className="text-lg font-semibold text-gray-900 mb-2">{feature.title}</h3>
                            <p className="text-gray-600 leading-relaxed">{feature.description}</p>
                        </div>
                        );
                    })}
                </div>
                </div>
            </div>
        </section>

        {/* How It Works Section (CONTENIDO ORIGINAL RESTAURADO) */}
        <section className="py-20 px-4 sm:px-6 lg:px-8 bg-white/80 backdrop-blur-sm">
            <div className="max-w-7xl mx-auto">
            <div className="text-center mb-16">
                <h2 className="text-3xl font-bold text-gray-900 mb-4">
                How Knowledge Navigator Works
                </h2>
                <p className="text-lg text-gray-600 max-w-2xl mx-auto">
                Three simple steps to unlock the intelligence hidden in your documents
                </p>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
                <div className="relative text-center">
                <div className="w-16 h-16 bg-gradient-to-br from-blue-500 to-blue-700 rounded-full flex items-center justify-center mx-auto mb-6">
                    <FileText className="w-8 h-8 text-white" />
                </div>
                <h3 className="text-xl font-semibold text-gray-900 mb-3">1. Upload Documents</h3>
                <p className="text-gray-600">
                    Securely upload your PDF documents to our enterprise-grade platform
                </p>
                <div className="hidden md:block absolute top-8 left-full w-full h-0.5 bg-gradient-to-r from-blue-300 to-blue-200 transform translate-x-4 -translate-y-0.5"></div>
                </div>

                <div className="relative text-center">
                <div className="w-16 h-16 bg-gradient-to-br from-blue-500 to-blue-700 rounded-full flex items-center justify-center mx-auto mb-6">
                    <Brain className="w-8 h-8 text-white" />
                </div>
                <h3 className="text-xl font-semibold text-gray-900 mb-3">2. AI Processing</h3>
                <p className="text-gray-600">
                    Our RAG technology analyzes and indexes your documents for intelligent retrieval
                </p>
                <div className="hidden md:block absolute top-8 left-full w-full h-0.5 bg-gradient-to-r from-blue-300 to-blue-200 transform translate-x-4 -translate-y-0.5"></div>
                </div>

                <div className="text-center">
                <div className="w-16 h-16 bg-gradient-to-br from-blue-500 to-blue-700 rounded-full flex items-center justify-center mx-auto mb-6">
                    <MessageSquare className="w-8 h-8 text-white" />
                </div>
                <h3 className="text-xl font-semibold text-gray-900 mb-3">3. Chat & Discover</h3>
                <p className="text-gray-600">
                    Ask natural language questions and get precise answers from your documents
                </p>
                </div>
            </div>
            </div>
        </section>

        {/* Stats Section (CONTENIDO ORIGINAL RESTAURADO) */}
        <section className="py-16 px-4 sm:px-6 lg:px-8 bg-gradient-to-r from-blue-600 to-blue-800">
            <div className="max-w-7xl mx-auto">
            <div className="grid grid-cols-1 md:grid-cols-3 gap-8 text-center text-white">
                <div>
                <div className="text-4xl font-bold mb-2">99.9%</div>
                <div className="text-blue-100">Accuracy Rate</div>
                </div>
                <div>
                <div className="text-4xl font-bold mb-2">&lt;2s</div>
                <div className="text-blue-100">Average Response Time</div>
                </div>
                <div>
                <div className="text-4xl font-bold mb-2">500+</div>
                <div className="text-blue-100">Enterprise Customers</div>
                </div>
            </div>
            </div>
        </section>
      </div>
    </div>
  );
};

export default Home;

