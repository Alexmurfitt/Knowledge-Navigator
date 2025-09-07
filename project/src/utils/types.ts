export interface User {
  id: string;
  name: string;
  email: string;
}

export interface Document {
  id: string;
  name: string;
  status: 'uploaded' | 'processing' | 'error';
  uploadedAt: Date;
  size?: number;
}

export interface Message {
  id: string;
  type: 'user' | 'assistant';
  content: string;
  timestamp: Date;
  hasAdditionalInfo?: boolean;
  additionalInfo?: {
    sources?: string[];
    confidence?: number;
    processingTime?: string;
    internetUsed?: boolean;
  };
}

export interface ChatState {
  messages: Message[];
  internetEnabled: boolean;
  isTyping: boolean;
}