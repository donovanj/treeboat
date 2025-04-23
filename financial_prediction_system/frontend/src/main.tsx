import React from 'react';
import ReactDOM from 'react-dom/client';
import * as Sentry from "@sentry/react";
import { ClerkProvider } from '@clerk/clerk-react';
import App from './App';
import { ThemeProvider, createTheme as createMuiTheme } from '@mui/material/styles';
import CssBaseline from '@mui/material/CssBaseline';
import { GlobalSelectionProvider } from './context/GlobalSelectionContext';

// --- Mantine Imports ---
import { MantineProvider, createTheme as createMantineTheme } from '@mantine/core';
import '@mantine/core/styles.css'; // Import core Mantine styles
// ---------------------

// Initialize Sentry
Sentry.init({
  dsn: "https://db4c4d6534cbd03e3cda47a2f40eed30@o4509098273275904.ingest.us.sentry.io/4509199131738112",
  integrations: [
    Sentry.browserTracingIntegration(),
    Sentry.replayIntegration({
      // Additional SDK configuration goes in here, for example:
      maskAllText: true,
      blockAllMedia: true,
    }),
  ],
  // Performance Monitoring
  tracesSampleRate: 1.0, //  Capture 100% of the transactions
  // Set 'tracePropagationTargets' to control for which URLs distributed tracing should be enabled
  tracePropagationTargets: ["localhost", "^https:\/\/yourserver\.io\/api"],
  // Session Replay
  replaysSessionSampleRate: 0.1, // This sets the sample rate at 10%.
  replaysOnErrorSampleRate: 1.0, // If you're not already sampling the entire session, change the sample rate to 100% when sampling sessions where errors occur.
});

// Create a default MUI theme
const muiTheme = createMuiTheme();

// Optional: Create a basic Mantine theme if you want to customize later
const mantineTheme = createMantineTheme({
  /** Put Mantine theme override here */
});

const clerkPubKey = import.meta.env.VITE_CLERK_PUBLISHABLE_KEY;

if (!clerkPubKey) {
  throw new Error('Missing Clerk Publishable Key');
}

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <ClerkProvider publishableKey={clerkPubKey}>
      <MantineProvider defaultColorScheme="light" theme={mantineTheme}>
        <ThemeProvider theme={muiTheme}>
          <CssBaseline /> { /* Normalize CSS */ }
          <GlobalSelectionProvider>
            <App />
          </GlobalSelectionProvider>
        </ThemeProvider>
      </MantineProvider>
    </ClerkProvider>
  </React.StrictMode>,
);
