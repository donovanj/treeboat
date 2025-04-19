import React, { useState, useEffect, useCallback } from 'react';
import { useGlobalSelection } from '../context/GlobalSelectionContext';
import { Typography, Paper } from '@mui/material';
import Plot from 'react-plotly.js';
import DateRangePicker from '../components/DateRangePicker';
import StockAutocomplete from '../components/StockAutocomplete';

const EDA: React.FC = () => {
  const { stock, setStock, startDate, setStartDate, endDate, setEndDate } = useGlobalSelection();
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [trendlinesFig, setTrendlinesFig] = useState<any>(null);
  const [ecdfFig, setEcdfFig] = useState<any>(null);
  const [histFig, setHistFig] = useState<any>(null);

  const fetchEDAData = useCallback(() => {
    setLoading(true);
    setError(null);
    let url = '/api/eda';
    const params = [];
    if (stock) params.push(`symbol=${encodeURIComponent(stock)}`);
    if (startDate) params.push(`start=${encodeURIComponent(startDate)}`);
    if (endDate) params.push(`end=${encodeURIComponent(endDate)}`);
    if (params.length) url += '?' + params.join('&');
    fetch(url)
      .then(res => {
        if (!res.ok) throw new Error('Failed to fetch EDA data');
        return res.json();
      })
      .then(data => {
        setTrendlinesFig(data.trendlines || null);
        setEcdfFig(data.ecdf || null);
        setHistFig(data.hist || null);
        setLoading(false);
      })
      .catch(err => {
        setError(err.message);
        setLoading(false);
      });
  }, [stock, startDate, endDate]);

  useEffect(() => {
    fetchEDAData();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [stock, startDate, endDate]);

  return (
    <Paper sx={{ p: 3 }}>
      <Typography variant="h4" gutterBottom>
        Exploratory Data Analysis (EDA)
      </Typography>
      <div style={{ margin: '20px 0' }}>
        <StockAutocomplete onSelect={setStock} />
        <DateRangePicker
          startDate={startDate}
          endDate={endDate}
          minDate="2020-01-02"
          maxDate={new Date().toISOString().slice(0, 10)}
          onChange={(start, end) => {
            setStartDate(start);
            setEndDate(end);
          }}
        />
        {loading && <Typography>Loading EDA data...</Typography>}
        {error && <Typography color="error">{error}</Typography>}
      </div>
      <div style={{marginTop: 40}}>
        <Typography variant="h5" gutterBottom>Linear and Non-Linear Trendlines</Typography>
        <Typography variant="body2" gutterBottom>
          Apply directly to raw price data to identify overall market direction.<br/>
          Help determine appropriate detrending methods before feature engineering.<br/>
          <b>Example:</b> Fitting exponential vs. linear trends to price series to understand growth patterns.
        </Typography>
        <Plot data={trendlinesFig?.data || []} layout={trendlinesFig?.layout || {title: 'Trendlines'}} style={{width: '100%', height: '300px'}} />
      </div>
      <div style={{marginTop: 40}}>
        <Typography variant="h5" gutterBottom>Empirical Cumulative Distribution Plots (ECDF)</Typography>
        <Typography variant="body2" gutterBottom>
          Excellent for raw returns (calculated from close prices).<br/>
          Helps identify the probability of returns falling below certain thresholds.<br/>
          Informs risk management decisions and feature engineering for risk metrics.
        </Typography>
        <Plot data={ecdfFig?.data || []} layout={ecdfFig?.layout || {title: 'ECDF'}} style={{width: '100%', height: '300px'}} />
      </div>
      <div style={{marginTop: 40}}>
        <Typography variant="h5" gutterBottom>Histograms & Distplots</Typography>
        <Typography variant="body2" gutterBottom>
          Use on raw daily returns to check for normality/non-normality.<br/>
          Apply to volume data to understand trading activity distribution.<br/>
          Identifies outliers and helps determine appropriate transformations (log, etc.).
        </Typography>
        <Plot data={histFig?.data || []} layout={histFig?.layout || {title: 'Histogram/Distplot'}} style={{width: '100%', height: '300px'}} />
      </div>
    </Paper>
  );
};

export default EDA;
