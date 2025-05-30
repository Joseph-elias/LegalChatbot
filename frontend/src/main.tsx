import React from 'react';               // (optional with new JSX, but OK)
import ReactDOM from 'react-dom/client'; // import the createRoot helper
import App from './App';                 // import your App component
 import './index.css';                  // import Tailwind (or your global styles)

ReactDOM.createRoot(
  document.getElementById('root') as HTMLElement
).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
