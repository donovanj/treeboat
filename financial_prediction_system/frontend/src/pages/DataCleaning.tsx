import React, { useEffect, useState, useCallback } from 'react';
import { useGlobalSelection } from '../context/GlobalSelectionContext';
import { 
  Typography, 
  Paper, 
  CircularProgress, 
  Alert, 
  Grid, 
  TableContainer, 
  Table, 
  TableHead, 
  TableRow, 
  TableCell, 
  TableBody 
} from '@mui/material';
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
  const [bandsFig, setBandsFig] = useState<any | null>(null);
  const [trendlinesFig, setTrendlinesFig] = useState<any | null>(null);
  const [describeStats, setDescribeStats] = useState<any | null>(null);
  const [missingHeatmapFig, setMissingHeatmapFig] = useState<any | null>(null);
  const [sampleSymbol, setSampleSymbol] = useState<string | null>(null);

  // Calculate default dates for 1 year of data
  const getDefaultDates = useCallback(() => {
    const today = new Date();
    const oneYearAgo = new Date();
    oneYearAgo.setFullYear(today.getFullYear() - 1);
    return {
      startDate: oneYearAgo.toISOString().slice(0, 10),
      endDate: today.toISOString().slice(0, 10)
    };
  }, []);

  useEffect(() => {
    // Set initial date range to 1 year if not already set
    if (!startDate || !endDate) {
      const { startDate: defaultStart, endDate: defaultEnd } = getDefaultDates();
      setStartDate(defaultStart);
      setEndDate(defaultEnd);
    }
  }, [getDefaultDates, startDate, endDate, setStartDate, setEndDate]);

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
          setBandsFig(data.bands_fig ? data.bands_fig : null);
          setTrendlinesFig(data.trendlines_fig ? data.trendlines_fig : null);
          setDescribeStats(data.describe_stats ? data.describe_stats : null);
          setMissingHeatmapFig(data.missing_heatmap_fig ? data.missing_heatmap_fig : null);
          
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
    if (startDate && endDate) {
      console.log(`DataCleaning: Fetching data for ${stock} from ${startDate} to ${endDate}`);
      fetchData();
    }
  }, [fetchData, startDate, endDate, stock]);

  return (
    <Paper sx={{ p: 3 }}>
      <Typography variant="h4" gutterBottom>
        Data Cleaning
      </Typography>
      <StockAutocomplete 
        // Remove the onSelect prop
        // onSelect={(symbol) => {
        //   setStock(symbol);
        // }} 
      />
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

          {/* --- 1. Basic Statistical Profiling (Moved Up) --- */}
          <Typography variant="h5" sx={{ mt: 4, mb: 1 }}>Basic Statistical Profiling</Typography>
          
          {/* Refactored Descriptive Statistics Table */}
          {describeStats && (
            <TableContainer component={Paper} sx={{ my: 3, maxWidth: 600 }}>
              <Typography variant="h6" sx={{ p: 2 }}>Descriptive Statistics</Typography>
              <Table size="small" aria-label="descriptive statistics table">
                <TableHead>
                  <TableRow sx={{ '& th': { fontWeight: 'bold' } }}>
                    <TableCell>Metric</TableCell>
                    <TableCell align="right">Close</TableCell>
                    <TableCell align="right">Volume</TableCell>
                    <TableCell align="right">Returns (%)</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {/* Explicitly define order and formatting */}
                  {['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max'].map((stat) => {
                    const formatValue = (value: any, decimals: number = 4, isPercent: boolean = false) => {
                      if (value === null || typeof value === 'undefined') return 'N/A';
                      const num = Number(value);
                      if (isNaN(num)) return String(value); // Return original if not a number
                      
                      // Use toLocaleString for large numbers (like count, volume)
                      if (['count', 'min', 'max'].includes(stat) && Math.abs(num) >= 1000 && !isPercent) {
                         return num.toLocaleString(undefined, { maximumFractionDigits: decimals });
                      }
                      // Handle percentages
                      if (isPercent) {
                        return (num * 100).toFixed(decimals > 0 ? decimals - 2 : 2); // Show returns as percentages
                      }
                      return num.toFixed(decimals);
                    };
                    
                    return (
                      <TableRow key={stat} sx={{ '&:last-child td, &:last-child th': { border: 0 } }}>
                        <TableCell component="th" scope="row">{stat}</TableCell>
                        <TableCell align="right">{formatValue(describeStats.close?.[stat], 4)}</TableCell>
                        <TableCell align="right">{formatValue(describeStats.volume?.[stat], 0)}</TableCell> 
                        <TableCell align="right">{formatValue(describeStats.returns?.[stat], 4, true)}</TableCell>
                      </TableRow>
                    );
                  })}
                </TableBody>
              </Table>
            </TableContainer>
          )}
          
          {/* Missing Data Heatmap */}
          {missingHeatmapFig && missingHeatmapFig.data && missingHeatmapFig.layout && (
            <div style={{ marginBottom: 24 }}>
              <Typography variant="h6">Missing Data Heatmap</Typography>
              <Plot 
                data={missingHeatmapFig.data} 
                layout={missingHeatmapFig.layout} 
                config={{ responsive: true }} 
                style={{ width: '100%', height: '400px' }}
              />
            </div>
          )}
          
          {/* --- 2. Time Series Exploration (Was 1) --- */}
          <Typography variant="h5" sx={{ mt: 4, mb: 1 }}>Time Series Exploration</Typography>
          {plotlyFig && plotlyFig.data && plotlyFig.layout && (
            <Plot 
              data={plotlyFig.data}
              layout={plotlyFig.layout}
              config={{ responsive: true }} 
              style={{ width: '100%', height: '550px' }}
            />
          )}
          
          {/* Convert line charts to candlestick */}
          {bandsFig && bandsFig.data && bandsFig.layout && (
            <Plot 
              data={bandsFig.data}
              layout={bandsFig.layout}
              config={{ responsive: true }} 
              style={{ width: '100%', height: '550px' }}
            />
          )}
          
          {trendlinesFig && trendlinesFig.data && trendlinesFig.layout && (
            <Plot 
              data={trendlinesFig.data}
              layout={trendlinesFig.layout}
              config={{ responsive: true }} 
              style={{ width: '100%', height: '550px' }}
            />
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
