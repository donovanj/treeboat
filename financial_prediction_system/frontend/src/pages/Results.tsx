import React from 'react';
import { Typography, Paper } from '@mui/material';

const Results: React.FC = () => (
  <Paper sx={{ p: 3 }}>
    <Typography variant="h4" gutterBottom>
      Results Visualization
    </Typography>
    {/* TODO: Visualize metrics, confusion matrix, feature importance, etc. */}
    <Typography>
      View model results and insights here.
    </Typography>
  </Paper>
);

export default Results;
