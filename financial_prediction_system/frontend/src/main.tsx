import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './App';
import { ThemeProvider, createTheme as createMuiTheme } from '@mui/material/styles';
import CssBaseline from '@mui/material/CssBaseline';
import { GlobalSelectionProvider } from './context/GlobalSelectionContext';
import { StockFeatureProvider } from './context/StockFeatureContext';

// --- Mantine Imports ---
import { MantineProvider, createTheme as createMantineTheme } from '@mantine/core';
import '@mantine/core/styles.css'; // Import core Mantine styles
// ---------------------

// Create a default MUI theme
const muiTheme = createMuiTheme();

// Optional: Create a basic Mantine theme if you want to customize later
const mantineTheme = createMantineTheme({
  /** Put Mantine theme override here */
});

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    {/* Configure MantineProvider - theme prop is optional for defaults */}
    <MantineProvider defaultColorScheme="light" theme={mantineTheme}>
      {/* Provide the default MUI theme */}
      <ThemeProvider theme={muiTheme}>
        <CssBaseline /> { /* Normalize CSS */ }
        <GlobalSelectionProvider>
          <StockFeatureProvider>
            <App />
          </StockFeatureProvider>
        </GlobalSelectionProvider>
      </ThemeProvider>
    </MantineProvider>
  </React.StrictMode>,
);
