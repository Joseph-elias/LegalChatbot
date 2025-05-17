import React, { useRef, useEffect } from 'react';
import { Scale } from 'lucide-react';
import ChatMessage from './ChatMessage';
import QuestionInput from './QuestionInput';
import { useMessageContext } from '../context/MessageContext';

const ChatInterface: React.FC = () => {
  const { messages, isTyping } = useMessageContext();
  const messagesEndRef = useRef<HTMLDivElement>(null);

  // Auto-scroll to bottom when new messages come in
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, isTyping]);

  return (
    <div className="max-w-4xl mx-auto">
      <div className="bg-white dark:bg-gray-800 rounded-lg shadow-lg overflow-hidden">
        {/* Welcome header */}
        <div className="bg-gradient-to-r from-blue-800 to-indigo-900 text-white p-6">
          <h2 className="text-2xl font-bold mb-2">مرحبا بك في المساعد القانوني اللبناني</h2>
          <h3 className="text-xl">Welcome to your Lebanese Legal Assistant</h3>
          <p className="mt-3 text-blue-100">
            Ask me any legal questions about Lebanese law, and I'll provide helpful information and relevant legal articles.
          </p>
        </div>
        
        {/* Chat messages container */}
        <div className="p-4 md:p-6 space-y-4 min-h-[300px] max-h-[60vh] overflow-y-auto law-pattern">
          {messages.length === 0 ? (
            <div className="flex flex-col items-center justify-center h-40 text-center p-4">
              <Scale className="h-12 w-12 text-gray-300 dark:text-gray-600 mb-4" />
              <p className="text-gray-500 dark:text-gray-400">
                Start by asking a question about Lebanese law
              </p>
            </div>
          ) : (
            messages.map((message, index) => (
              <ChatMessage
                key={index}
                message={message.text}
                isUser={message.isUser}
                timestamp={message.timestamp}
                sources={message.sources}
              />
            ))
          )}
          
          {isTyping && (
            <div className="flex items-start space-x-2 animate-fade-in">
              <div className="flex-shrink-0 bg-blue-100 dark:bg-blue-900 rounded-full p-2">
                <Scale className="h-6 w-6 text-blue-600 dark:text-blue-300" />
              </div>
              <div className="p-3 bg-blue-100 dark:bg-blue-900 rounded-lg max-w-[80%]">
                <div className="flex space-x-2">
                  <div className="h-2 w-2 bg-blue-500 rounded-full animate-typing delay-100"></div>
                  <div className="h-2 w-2 bg-blue-500 rounded-full animate-typing delay-200"></div>
                  <div className="h-2 w-2 bg-blue-500 rounded-full animate-typing delay-300"></div>
                </div>
              </div>
            </div>
          )}
          
          <div ref={messagesEndRef} />
        </div>
        
        {/* Question input */}
        <div className="border-t border-gray-200 dark:border-gray-700">
          <QuestionInput />
        </div>
      </div>
    </div>
  );
};

export default ChatInterface;
