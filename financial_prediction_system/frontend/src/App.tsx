import React from 'react';
import { CssBaseline, ThemeProvider, createTheme } from '@mui/material';
import { BrowserRouter as Router, Routes, Route, Link } from 'react-router-dom';
import { Box, AppBar, Toolbar, Typography, Drawer, List, ListItem, ListItemText, Button } from '@mui/material';
import { SignInButton, SignUpButton, UserButton, useUser } from '@clerk/clerk-react';
import { ProtectedRoute } from './components/ProtectedRoute';
import Dashboard from './pages/Dashboard';
import DataCleaning from './pages/DataCleaning';
import EDA from './pages/EDA';

const drawerWidth = 220;
const theme = createTheme();

const navItems = [
  { text: 'Dashboard', path: '/' },
  { text: 'Data Cleaning', path: '/cleaning' },
  { text: 'EDA', path: '/eda' }
];

function App() {
  const { isSignedIn } = useUser();

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Router>
        <Box sx={{ display: 'flex' }}>
          <AppBar position="fixed" sx={{ zIndex: (theme) => theme.zIndex.drawer + 1 }}>
            <Toolbar sx={{ display: 'flex', justifyContent: 'space-between' }}>
              <Typography variant="h6" noWrap component="div">
                Financial Prediction ML Workflow
              </Typography>
              <Box sx={{ display: 'flex', gap: 2, alignItems: 'center' }}>
                {!isSignedIn ? (
                  <>
                    <SignInButton mode="modal">
                      <Button color="inherit">Sign In</Button>
                    </SignInButton>
                    <SignUpButton mode="modal">
                      <Button color="inherit">Sign Up</Button>
                    </SignUpButton>
                  </>
                ) : (
                  <UserButton afterSignOutUrl="/" />
                )}
              </Box>
            </Toolbar>
          </AppBar>
          {isSignedIn && (
            <Drawer
              variant="permanent"
              sx={{
                width: drawerWidth,
                flexShrink: 0,
                [`& .MuiDrawer-paper`]: { width: drawerWidth, boxSizing: 'border-box' },
              }}
            >
              <Toolbar />
              <List>
                {navItems.map((item) => (
                  <ListItem button key={item.text} component={Link} to={item.path}>
                    <ListItemText primary={item.text} />
                  </ListItem>
                ))}
              </List>
            </Drawer>
          )}
          <Box component="main" sx={{ flexGrow: 1, pt: 1, ml: isSignedIn ? 0 : 0 }}>
            <Toolbar />
            <Routes>
              <Route path="/" element={
                !isSignedIn ? (
                  <Box sx={{ p: 4, textAlign: 'center' }}>
                    <Typography variant="h5" gutterBottom>
                      Welcome to Financial Prediction ML Workflow
                    </Typography>
                    <Typography>
                      Please sign in to access the application.
                    </Typography>
                  </Box>
                ) : (
                  <ProtectedRoute>
                    <Dashboard />
                  </ProtectedRoute>
                )
              } />
              <Route path="/cleaning" element={
                <ProtectedRoute>
                  <DataCleaning />
                </ProtectedRoute>
              } />
              <Route path="/eda" element={
                <ProtectedRoute>
                  <EDA />
                </ProtectedRoute>
              } />
            </Routes>
          </Box>
        </Box>
      </Router>
    </ThemeProvider>
  );
}

export default App;
