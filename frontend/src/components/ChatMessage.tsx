import React, { useState } from 'react';
import { User, Scale, BookOpen, ChevronDown, ChevronUp, ExternalLink } from 'lucide-react';

interface Source {
  title: string;
  excerpt: string;
  url: string;
}

interface ChatMessageProps {
  message: string;
  isUser: boolean;
  timestamp: Date;
  sources?: Source[];
}

const ChatMessage: React.FC<ChatMessageProps> = ({ message, isUser, timestamp, sources }) => {
  const [showSources, setShowSources] = useState(false);
  
  const formattedTime = timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });

  return (
    <div className={`flex animate-fade-in ${isUser ? 'justify-end' : 'justify-start'}`}>
      <div className={`max-w-[80%] ${isUser ? 'order-1' : 'order-2'}`}>
        <div className="flex items-start space-x-2">
          <div className={`flex-shrink-0 ${isUser ? 'order-2 ml-2' : 'order-1 mr-2'}`}>
            <div className={`rounded-full p-2 ${
              isUser 
                ? 'bg-amber-100 dark:bg-amber-900' 
                : 'bg-blue-100 dark:bg-blue-900'
            }`}>
              {isUser 
                ? <User className="h-5 w-5 text-amber-600 dark:text-amber-300" /> 
                : <Scale className="h-5 w-5 text-blue-600 dark:text-blue-300" />
              }
            </div>
          </div>
          
          <div className={`${
            isUser 
              ? 'order-1 bg-amber-50 dark:bg-amber-900/60 text-gray-800 dark:text-gray-100' 
              : 'order-2 bg-white dark:bg-gray-800 text-gray-800 dark:text-gray-100 border border-gray-200 dark:border-gray-700'
            } p-3 rounded-lg shadow-sm`}
          >
            <div className="whitespace-pre-wrap">{message}</div>
            
            {!isUser && sources && sources.length > 0 && (
              <div className="mt-3">
                <button
                  onClick={() => setShowSources(!showSources)}
                  className="flex items-center text-xs text-blue-600 dark:text-blue-400 font-medium hover:underline focus:outline-none"
                >
                  {showSources ? (
                    <>
                      <ChevronUp className="h-3 w-3 mr-1" />
                      Hide sources
                    </>
                  ) : (
                    <>
                      <ChevronDown className="h-3 w-3 mr-1" />
                      Show {sources.length} legal source{sources.length > 1 ? 's' : ''}
                    </>
                  )}
                </button>
                
                {showSources && (
                  <div className="mt-2 space-y-2 text-sm border-t border-gray-200 dark:border-gray-700 pt-2">
                    {sources.map((source, index) => (
                      <div key={index} className="bg-gray-50 dark:bg-gray-900 p-2 rounded">
                        <div className="flex items-start">
                          <BookOpen className="h-4 w-4 text-gray-500 dark:text-gray-400 mt-1 mr-2 flex-shrink-0" />
                          <div>
                            <div className="font-medium">{source.title}</div>
                            <p className="text-gray-600 dark:text-gray-300 text-xs mt-1">{source.excerpt}</p>
                            <a
                              href={source.url}
                              target="_blank"
                              rel="noopener noreferrer"
                              className="text-xs inline-flex items-center text-blue-600 dark:text-blue-400 hover:underline mt-1"
                            >
                              Read full article <ExternalLink className="h-3 w-3 ml-1" />
                            </a>
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            )}
            
            <div className="text-xs text-gray-500 dark:text-gray-400 mt-1 text-right">
              {formattedTime}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ChatMessage;