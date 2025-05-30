import React, { useState } from 'react';
import { SendHorizonal, Mic, Search } from 'lucide-react';
import { useMessageContext } from '../context/MessageContext';

const QuestionInput: React.FC = () => {
  const [question, setQuestion] = useState('');
  const { addMessage, setTyping } = useMessageContext();
  
  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!question.trim()) return;

    addMessage(question, true);
    setQuestion('');
    setTyping(true);

    try {
      
      const res = await fetch(`${import.meta.env.VITE_API_BASE}/search`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: question, top_k: 3, alpha: 0.6 }),
      });

      if (!res.ok) {
        throw new Error(`Server returned status ${res.status}`);
      }

      const { answer, sources } = await res.json();

      addMessage(answer, false, sources);
    } catch (err: any) {
      console.error('Fetch error:', err);
      addMessage(
        `عذراً، حدث خطأ أثناء الاتصال بالخادم:\n${err.message || err}`,
        false
      );
    } finally {
      setTyping(false);
    }
  };

  return (
    <form onSubmit={handleSubmit} className="p-4 bg-white dark:bg-gray-800">
      <div className="flex items-center space-x-2">
        <button
          type="button"
          className="p-2 text-gray-500 rounded-full hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors"
          aria-label="Search legal documents"
        >
          <Search className="h-5 w-5" />
        </button>

        <div className="flex-grow relative">
          <input
            type="text"
            placeholder="Type your legal question..."
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            className="w-full p-3 pr-10 rounded-lg border border-gray-300 dark:border-gray-600 bg-gray-50 dark:bg-gray-700 dark:text-white placeholder-gray-500 dark:placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-blue-500 dark:focus:ring-blue-600 transition-all"
          />
        </div>

        <button
          type="button"
          className="p-2 text-gray-500 rounded-full hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors"
          aria-label="Voice input"
        >
          <Mic className="h-5 w-5" />
        </button>

        <button
          type="submit"
          disabled={!question.trim()}
          className={`p-3 rounded-full ${
            question.trim()
              ? 'bg-blue-600 text-white hover:bg-blue-700'
              : 'bg-gray-200 text-gray-500 dark:bg-gray-700 dark:text-gray-400 cursor-not-allowed'
          } transition-colors`}
          aria-label="Send message"
        >
          <SendHorizonal className="h-5 w-5" />
        </button>
      </div>
    </form>
  );
};

export default QuestionInput;
