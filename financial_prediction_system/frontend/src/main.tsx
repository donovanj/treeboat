import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './App';

import { GlobalSelectionProvider } from './context/GlobalSelectionContext';

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <GlobalSelectionProvider>
      <App />
    </GlobalSelectionProvider>
  </React.StrictMode>
);
