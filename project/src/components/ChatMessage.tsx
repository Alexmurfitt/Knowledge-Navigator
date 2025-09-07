import React from 'react';
import { User, FileText, Info } from 'lucide-react';
import { Message } from '../utils/types';

interface ChatMessageProps {
  message: Message;
  onShowInfo?: () => void;
}

const ChatMessage: React.FC<ChatMessageProps> = ({ message, onShowInfo }) => {
  const isUser = message.type === 'user';

  return (
    <div className={`flex items-start space-x-3 ${isUser ? 'flex-row-reverse space-x-reverse' : ''}`}>
      {/* Avatar */}
      <div className={`w-8 h-8 rounded-full flex items-center justify-center ${
        isUser ? 'bg-gray-600' : 'bg-blue-600'
      }`}>
        {isUser ? (
          <User className="w-4 h-4 text-white" />
        ) : (
          <FileText className="w-4 h-4 text-white" />
        )}
      </div>

      {/* Message Content */}
      <div className={`flex-1 max-w-3xl ${isUser ? 'text-right' : ''}`}>
        <div className="relative">
          <div className={`inline-block p-4 rounded-xl shadow-sm ${
            isUser 
              ? 'bg-blue-600 text-white' 
              : 'bg-white text-gray-900 border border-gray-200'
          }`}>
            <p className="leading-relaxed whitespace-pre-wrap">{message.content}</p>
            
            {/* Info Button for Assistant Messages */}
            {!isUser && message.hasAdditionalInfo && onShowInfo && (
              <button
                onClick={onShowInfo}
                className="absolute -top-2 -right-2 w-6 h-6 bg-blue-600 hover:bg-blue-700 text-white rounded-full flex items-center justify-center transition-all duration-200 hover:scale-110 shadow-md"
                title="View additional information"
              >
                <Info className="w-3 h-3" />
              </button>
            )}
          </div>
        </div>
        
        {/* Timestamp */}
        <div className={`text-xs text-gray-500 mt-2 ${isUser ? 'text-right' : ''}`}>
          {message.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
        </div>
      </div>
    </div>
  );
};

export default ChatMessage;