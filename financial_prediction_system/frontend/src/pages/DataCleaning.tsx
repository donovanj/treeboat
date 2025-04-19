import React, { useEffect, useState, useCallback } from 'react';
import { useGlobalSelection } from '../context/GlobalSelectionContext';
import { Typography, Paper, CircularProgress, Alert } from '@mui/material';
import StockAutocomplete from '../components/StockAutocomplete';
import Plot from 'react-plotly.js';
import DateRangePicker from '../components/DateRangePicker';

const DataCleaning: React.FC = () => {
  const { stock, setStock, startDate, setStartDate, endDate, setEndDate } = useGlobalSelection();
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [summary, setSummary] = useState<any>({});
  const [issues, setIssues] = useState<any>({});
  const [priceIssues, setPriceIssues] = useState<any>({});
  const [plotlyFig, setPlotlyFig] = useState<any | null>(null);
  const [ohlcvFig, setOhlcvFig] = useState<any | null>(null);
  const [bandsFig, setBandsFig] = useState<any | null>(null);
  const [trendlinesFig, setTrendlinesFig] = useState<any | null>(null);
  const [describeStats, setDescribeStats] = useState<any | null>(null);
  const [missingHeatmapFig, setMissingHeatmapFig] = useState<any | null>(null);
  const [boxViolinFig, setBoxViolinFig] = useState<any | null>(null);
  const [sampleSymbol, setSampleSymbol] = useState<string | null>(null);

  const fetchData = useCallback(() => {
    setLoading(true);
    setError(null);
    let url = '/api/data-cleaning';
    const params = [];
    if (stock) params.push(`symbol=${encodeURIComponent(stock)}`);
    if (startDate) params.push(`start=${encodeURIComponent(startDate)}`);
    if (endDate) params.push(`end=${encodeURIComponent(endDate)}`);
    if (params.length) url += '?' + params.join('&');
    fetch(url)
      .then(res => {
        if (!res.ok) throw new Error('Failed to fetch data');
        return res.json();
      })
      .then(data => {
        console.log('API response:', data);
        if (data.success) {
          setSummary(data.summary || {});
          setIssues(data.issues || {});
          setPriceIssues(data.price_issues || {});
          setPlotlyFig(data.plotly_fig ? data.plotly_fig : null);
          setOhlcvFig(data.ohlcv_fig ? data.ohlcv_fig : null);
          setBandsFig(data.bands_fig ? data.bands_fig : null);
          setTrendlinesFig(data.trendlines_fig ? data.trendlines_fig : null);
          setDescribeStats(data.describe_stats ? data.describe_stats : null);
          setMissingHeatmapFig(data.missing_heatmap_fig ? data.missing_heatmap_fig : null);
          setBoxViolinFig(data.box_violin_fig ? data.box_violin_fig : null);
          setSampleSymbol(data.sample_symbol || null);
        } else {
          setError(data.error || 'Unknown error');
        }
        setLoading(false);
      })
      .catch(err => {
        setError(err.message);
        setLoading(false);
      });
  }, [stock, startDate, endDate]);

  useEffect(() => {
    fetchData();
  }, [fetchData]);

  return (
    <Paper sx={{ p: 3 }}>
      <Typography variant="h4" gutterBottom>
        Data Cleaning
      </Typography>
      <StockAutocomplete onSelect={(symbol) => {
        setStock(symbol);
      }} />
      {/* Date Range Picker */}
      <div style={{ margin: '20px 0' }}>
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
      </div>
      {loading && <CircularProgress />}
      {error && <Alert severity="error">{error}</Alert>}
      {!loading && !error && (
        <>

          {sampleSymbol && (
            <Typography variant="subtitle2">Sample Symbol: <b>{sampleSymbol}</b></Typography>
          )}
          {/* --- 1. Initial Time Series Exploration (Candlestick Chart) --- */}
          <Typography variant="h5" sx={{ mt: 4, mb: 1 }}>1. Initial Time Series Exploration (Candlestick Chart)</Typography>
          {plotlyFig && plotlyFig.data && plotlyFig.layout && (
            <Plot data={plotlyFig.data} layout={plotlyFig.layout} config={{ responsive: true }} style={{ width: '100%', height: '400px' }} />
          )}
          {bandsFig && bandsFig.data && bandsFig.layout && (
            <Plot data={bandsFig.data} layout={bandsFig.layout} config={{ responsive: true }} style={{ width: '100%', height: '400px' }} />
          )}
          {trendlinesFig && trendlinesFig.data && trendlinesFig.layout && (
            <Plot data={trendlinesFig.data} layout={trendlinesFig.layout} config={{ responsive: true }} style={{ width: '100%', height: '400px' }} />
          )}

          {/* --- 2. Basic Statistical Profiling --- */}
          <Typography variant="h5" sx={{ mt: 4, mb: 1 }}>2. Basic Statistical Profiling</Typography>
          {describeStats && (
            <div style={{ overflowX: 'auto', marginBottom: 24 }}>
              <Typography variant="h6">Descriptive Statistics</Typography>
              <table style={{ borderCollapse: 'collapse', minWidth: 400 }}>
                <thead>
                  <tr>
                    <th style={{ border: '1px solid #ccc', padding: 4 }}>Metric</th>
                    <th style={{ border: '1px solid #ccc', padding: 4 }}>Close</th>
                    <th style={{ border: '1px solid #ccc', padding: 4 }}>Volume</th>
                    <th style={{ border: '1px solid #ccc', padding: 4 }}>Returns</th>
                  </tr>
                </thead>
                <tbody>
                  {Object.keys(describeStats.close).map((stat) => (
                    <tr key={stat}>
                      <td style={{ border: '1px solid #ccc', padding: 4 }}>{stat}</td>
                      <td style={{ border: '1px solid #ccc', padding: 4 }}>{describeStats.close[stat]?.toFixed ? String(describeStats.close[stat].toFixed(4)) : String(describeStats.close[stat])}</td>
                      <td style={{ border: '1px solid #ccc', padding: 4 }}>{describeStats.volume[stat]?.toFixed ? String(describeStats.volume[stat].toFixed(2)) : String(describeStats.volume[stat])}</td>
                      <td style={{ border: '1px solid #ccc', padding: 4 }}>{describeStats.returns[stat]?.toFixed ? String(describeStats.returns[stat].toFixed(4)) : String(describeStats.returns[stat])}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
          {missingHeatmapFig && missingHeatmapFig.data && missingHeatmapFig.layout && (
            <div style={{ marginBottom: 24 }}>
              <Typography variant="h6">Missing Data Heatmap</Typography>
              <Plot data={missingHeatmapFig.data} layout={missingHeatmapFig.layout} config={{ responsive: true }} style={{ width: '100%', height: '300px' }} />
            </div>
          )}
          {boxViolinFig && boxViolinFig.data && boxViolinFig.layout && (
            <div style={{ marginBottom: 24 }}>
              <Typography variant="h6">Box & Violin Plots</Typography>
              <Plot data={boxViolinFig.data} layout={boxViolinFig.layout} config={{ responsive: true }} style={{ width: '100%', height: '400px' }} />
            </div>
          )}

          {/* --- Legacy and Table Sections --- */}
          <Typography variant="h6" sx={{ mt: 2 }}>Stocks Table Summary</Typography>
          <ul>
            {Object.entries(summary).map(([key, val]) => (
              <li key={key}><b>{key}:</b> {typeof val === 'string' || typeof val === 'number' ? val : JSON.stringify(val)}</li>
            ))}
          </ul>
          <Typography variant="h6" sx={{ mt: 2 }}>Stocks Table Issues</Typography>
          <ul>
            {Object.entries(issues).map(([key, val]) => (
              <li key={key}>
                <b>{key}:</b>{' '}
                {Array.isArray(val)
                  ? val.join(', ')
                  : typeof val === 'string' || typeof val === 'number'
                    ? val
                    : JSON.stringify(val)}
              </li>
            ))}
          </ul>
          <Typography variant="h6" sx={{ mt: 2 }}>Stock Prices Table Analysis</Typography>
          <ul>
            {Object.entries(priceIssues).map(([key, val]) => (
              <li key={key}><b>{key}:</b> {Array.isArray(val) ? val.join(', ') : JSON.stringify(val)}</li>
            ))}
          </ul>


        </>
      )}
    </Paper>
  );
};

export default DataCleaning;
