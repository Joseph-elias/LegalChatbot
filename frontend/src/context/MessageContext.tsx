import React, { createContext, useContext, useState, ReactNode } from 'react';

interface Source {
  title: string;
  excerpt: string;
  url: string;
}

interface Message {
  text: string;
  isUser: boolean;
  timestamp: Date;
  sources?: Source[];
}

interface MessageContextType {
  messages: Message[];
  isTyping: boolean;
  addMessage: (text: string, isUser: boolean, sources?: Source[]) => void;
  setTyping: (isTyping: boolean) => void;
}

const MessageContext = createContext<MessageContextType | undefined>(undefined);

export const MessageProvider: React.FC<{ children: ReactNode }> = ({ children }) => {
  const [messages, setMessages] = useState<Message[]>([]);
  const [isTyping, setIsTyping] = useState(false);

  const addMessage = (text: string, isUser: boolean, sources?: Source[]) => {
    const newMessage: Message = {
      text,
      isUser,
      timestamp: new Date(),
      sources,
    };
    
    setMessages((prevMessages) => [...prevMessages, newMessage]);
  };

  return (
    <MessageContext.Provider value={{ messages, isTyping, addMessage, setTyping: setIsTyping }}>
      {children}
    </MessageContext.Provider>
  );
};

export const useMessageContext = (): MessageContextType => {
  const context = useContext(MessageContext);
  if (context === undefined) {
    throw new Error('useMessageContext must be used within a MessageProvider');
  }
  return context;
};