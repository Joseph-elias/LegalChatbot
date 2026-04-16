import React from 'react';
import { Info, Linkedin, Github, Globe } from 'lucide-react';

interface FooterProps {
  compact?: boolean;
}

const Footer: React.FC<FooterProps> = ({ compact = false }) => {
  if (compact) {
    return (
      <footer className="bg-gray-800 text-gray-300 py-3">
        <div className="container mx-auto px-4">
          <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-3">
            <div className="flex items-center space-x-2">
              <img src="/logo.png" alt="Lebanese Legal RAG Assistant" className="h-8 w-8 rounded object-cover" />
              <span className="text-sm text-white font-semibold">LebLegal</span>
              <span className="text-xs text-gray-400">AI legal assistant for Lebanese law</span>
            </div>
            <div className="flex items-center gap-3">
              <a
                href="https://www.linkedin.com/in/joseph-elias-al-khoury-0a54a8239/"
                target="_blank"
                rel="noopener noreferrer"
                className="text-gray-300 hover:text-amber-400 transition-colors"
                aria-label="LinkedIn Profile"
              >
                <Linkedin className="h-4 w-4" />
              </a>
              <a
                href="https://github.com/Joseph-elias"
                target="_blank"
                rel="noopener noreferrer"
                className="text-gray-300 hover:text-amber-400 transition-colors"
                aria-label="GitHub Profile"
              >
                <Github className="h-4 w-4" />
              </a>
              <a
                href="https://joseph-elias.github.io/Portfolio-joseph/"
                target="_blank"
                rel="noopener noreferrer"
                className="text-gray-300 hover:text-amber-400 transition-colors"
                aria-label="Portfolio Website"
              >
                <Globe className="h-4 w-4" />
              </a>
            </div>
          </div>
          <div className="border-t border-gray-700 mt-3 pt-2 text-center text-xs text-gray-400">
            (c) {new Date().getFullYear()} LebLegal. All rights reserved.
          </div>
        </div>
      </footer>
    );
  }

  return (
    <footer className="bg-gray-800 text-gray-300 py-6">
      <div className="container mx-auto px-4">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
          <div>
            <div className="flex items-center space-x-2 mb-4">
              <img src="/logo.png" alt="Lebanese Legal RAG Assistant" className="h-10 w-10 rounded object-cover" />
              <h2 className="text-xl font-bold text-white">LebLegal</h2>
            </div>
            <p className="text-sm">
              Your trusted AI-powered legal assistant for navigating Lebanese law.
            </p>
          </div>

          <div>
            <h3 className="text-lg font-semibold mb-4 text-white">Disclaimer</h3>
            <p className="text-sm">
              This tool provides legal information for educational purposes only.
              It is not a substitute for professional legal advice. Always consult
              with a qualified attorney for your specific legal matters.
            </p>
          </div>

          <div>
            <h3 className="text-lg font-semibold mb-4 text-white">Contact</h3>
            <p className="text-sm mb-4">
              For questions or feedback, please contact us at:
              <a href="mailto:josepheliaskh@gmail.com" className="text-amber-400 hover:text-amber-300 transition-colors block mt-1">
                josepheliaskh@gmail.com
              </a>
            </p>
            <div className="flex space-x-4">
              <a
                href="https://www.linkedin.com/in/joseph-elias-al-khoury-0a54a8239/"
                target="_blank"
                rel="noopener noreferrer"
                className="text-gray-300 hover:text-amber-400 transition-colors"
                aria-label="LinkedIn Profile"
              >
                <Linkedin className="h-5 w-5" />
              </a>
              <a
                href="https://github.com/Joseph-elias"
                target="_blank"
                rel="noopener noreferrer"
                className="text-gray-300 hover:text-amber-400 transition-colors"
                aria-label="GitHub Profile"
              >
                <Github className="h-5 w-5" />
              </a>
              <a
                href="https://joseph-elias.github.io/Portfolio-joseph/"
                target="_blank"
                rel="noopener noreferrer"
                className="text-gray-300 hover:text-amber-400 transition-colors"
                aria-label="Portfolio Website"
              >
                <Globe className="h-5 w-5" />
              </a>
            </div>
          </div>
        </div>

        <div className="border-t border-gray-700 mt-8 pt-6 text-center text-sm">
          <div className="flex items-center justify-center space-x-1 mb-2">
            <Info className="h-4 w-4" />
            <p>
              Uses semantic search technology to provide information based on Lebanese legal codes.
            </p>
          </div>
          <p>(c) {new Date().getFullYear()} LebLegal. All rights reserved.</p>
        </div>
      </div>
    </footer>
  );
};

export default Footer;
