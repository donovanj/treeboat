import React from 'react';
import { Typography, Paper } from '@mui/material';

const Dashboard: React.FC = () => (
  <Paper sx={{ p: 3 }}>
    <Typography variant="h4" gutterBottom>
      Data Overview & EDA
    </Typography>
    {/* TODO: Add data overview, tables, distributions, time series, correlations, filters */}
    <Typography>
      Visualize and explore your SEC, stock, and macro data here.
    </Typography>
  </Paper>
);

export default Dashboard;
