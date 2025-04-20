import React from 'react';
import { CssBaseline, ThemeProvider, createTheme } from '@mui/material';
import { BrowserRouter as Router, Routes, Route, Link } from 'react-router-dom';
import { Box, AppBar, Toolbar, Typography, Drawer, List, ListItem, ListItemText } from '@mui/material';
import Dashboard from './pages/Dashboard';
import DataCleaning from './pages/DataCleaning';
import FeatureEngineering from './pages/FeatureEngineering';
import TargetConstruction from './pages/TargetConstruction';
import ModelTraining from './pages/ModelTraining';
import Results from './pages/Results';
import EDA from './pages/EDA';

const drawerWidth = 220;
const theme = createTheme();

const navItems = [
  { text: 'Dashboard', path: '/' },
  { text: 'Data Cleaning', path: '/cleaning' },
  { text: 'EDA', path: '/eda' },
  { text: 'Feature Engineering', path: '/features' },
  { text: 'Target Construction', path: '/target' },
  { text: 'Model Training', path: '/training' },
  { text: 'Results', path: '/results' }
];

function App() {
  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Router>
        <Box sx={{ display: 'flex' }}>
          <AppBar position="fixed" sx={{ zIndex: (theme) => theme.zIndex.drawer + 1 }}>
            <Toolbar>
              <Typography variant="h6" noWrap component="div">
                Financial Prediction ML Workflow
              </Typography>
            </Toolbar>
          </AppBar>
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
          <Box component="main" sx={{ flexGrow: 1, pt: 1, ml: 0 }}>
            <Toolbar />
            <Routes>
              <Route path="/" element={<Dashboard />} />
              <Route path="/cleaning" element={<DataCleaning />} />
              <Route path="/eda" element={<EDA />} />
              <Route path="/features" element={<FeatureEngineering />} />
              <Route path="/target" element={<TargetConstruction />} />
              <Route path="/training" element={<ModelTraining />} />
              <Route path="/results" element={<Results />} />
            </Routes>
          </Box>
        </Box>
      </Router>
    </ThemeProvider>
  );
}

export default App;
