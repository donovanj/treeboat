import React from 'react';
import { Typography, Paper } from '@mui/material';

const ModelTraining: React.FC = () => (
  <Paper sx={{ p: 3 }}>
    <Typography variant="h4" gutterBottom>
      Model Selection & Training
    </Typography>
    {/* TODO: UI for model selection, parameter config, feature selection, scaling, training trigger */}
    <Typography>
      Select model type, configure, and train here.
    </Typography>
  </Paper>
);

export default ModelTraining;
