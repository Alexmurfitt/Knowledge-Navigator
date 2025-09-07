import React from 'react';
import Header from './components/Header';
import Home from './pages/Home';
import Documents from './pages/Documents';
import Chat from './pages/Chat';
import { AuthProvider, useAuth } from './hooks/useAuth';
import { NavigationProvider, useNavigation } from './hooks/useNavigation';
import './styles/animations.css';

function AppContent() {
  const { currentPage } = useNavigation();
  const { isAuthenticated } = useAuth();

  const renderPage = () => {
    switch (currentPage) {
      case 'home':
        return <Home />;
      case 'documents':
        return isAuthenticated ? <Documents /> : <Home />;
      case 'chat':
        return isAuthenticated ? <Chat /> : <Home />;
      default:
        return <Home />;
    }
  };

  return (
    <div className="min-h-screen bg-gray-50">
      <Header />
      <main className="pt-16">
        {renderPage()}
      </main>
    </div>
  );
}

function App() {
  return (
    <AuthProvider>
      <NavigationProvider>
        <AppContent />
      </NavigationProvider>
    </AuthProvider>
  );
}

export default App;