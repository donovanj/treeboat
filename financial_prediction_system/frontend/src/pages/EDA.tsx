import React, { useState, useEffect, useCallback } from 'react';
import { useGlobalSelection } from '../context/GlobalSelectionContext';
import { Typography, Paper, CircularProgress, Alert, Grid, Tabs, Tab, Box, Button } from '@mui/material';
import Plot from 'react-plotly.js';
import DateRangePicker from '../components/DateRangePicker';
import StockAutocomplete from '../components/StockAutocomplete';

interface TabPanelProps {
  children?: React.ReactNode;
  index: number;
  value: number;
}

function TabPanel(props: TabPanelProps) {
  const { children, value, index, ...other } = props;

  return (
    <div
      role="tabpanel"
      hidden={value !== index}
      id={`eda-tabpanel-${index}`}
      aria-labelledby={`eda-tab-${index}`}
      {...other}
    >
      {value === index && (
        <Box sx={{ p: 3 }}>
          {children}
        </Box>
      )}
    </div>
  );
}

const EDA: React.FC = () => {
  const { stock, setStock, startDate, setStartDate, endDate, setEndDate } = useGlobalSelection();
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [tabValue, setTabValue] = useState(0);
  
  // Original plots
  const [trendlinesFig, setTrendlinesFig] = useState<any>(null);
  const [ecdfFig, setEcdfFig] = useState<any>(null);
  const [histFig, setHistFig] = useState<any>(null);
  
  // New OHLCV visualizations
  const [candlestickFig, setCandlestickFig] = useState<any>(null);
  
  // Price data visualizations
  const [returnsDistFig, setReturnsDistFig] = useState<any>(null);
  const [priceRangeBoxFig, setPriceRangeBoxFig] = useState<any>(null);
  const [priceRangeViolinFig, setPriceRangeViolinFig] = useState<any>(null);
  const [gapAnalysisBoxFig, setGapAnalysisBoxFig] = useState<any>(null);
  const [gapAnalysisViolinFig, setGapAnalysisViolinFig] = useState<any>(null);
  const [closePositioningBoxFig, setClosePositioningBoxFig] = useState<any>(null);
  const [closePositioningViolinFig, setClosePositioningViolinFig] = useState<any>(null);
  const [multitimeframeBoxFig, setMultitimeframeBoxFig] = useState<any>(null);
  const [multitimeframeViolinFig, setMultitimeframeViolinFig] = useState<any>(null);
  const [seasonalBoxFig, setSeasonalBoxFig] = useState<any>(null);
  const [seasonalViolinFig, setSeasonalViolinFig] = useState<any>(null);
  
  // Volume data visualizations
  const [volumeDistBoxFig, setVolumeDistBoxFig] = useState<any>(null);
  const [volumeDistViolinFig, setVolumeDistViolinFig] = useState<any>(null);
  const [relativeVolumeBoxFig, setRelativeVolumeBoxFig] = useState<any>(null);
  const [relativeVolumeViolinFig, setRelativeVolumeViolinFig] = useState<any>(null);
  const [priceVolumeBoxFig, setPriceVolumeBoxFig] = useState<any>(null);
  const [priceVolumeViolinFig, setPriceVolumeViolinFig] = useState<any>(null);
  const [volumePersistenceBoxFig, setVolumePersistenceBoxFig] = useState<any>(null);
  const [volumePersistenceViolinFig, setVolumePersistenceViolinFig] = useState<any>(null);
  const [volumeEventBoxFig, setVolumeEventBoxFig] = useState<any>(null);
  const [volumeEventViolinFig, setVolumeEventViolinFig] = useState<any>(null);
  const [volumePriceLevelBoxFig, setVolumePriceLevelBoxFig] = useState<any>(null);
  const [volumePriceLevelViolinFig, setVolumePriceLevelViolinFig] = useState<any>(null);

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

  const handleTabChange = (event: React.SyntheticEvent, newValue: number) => {
    setTabValue(newValue);
  };

  const fetchEDAData = useCallback(() => {
    setLoading(true);
    setError(null);
    let url = '/api/eda';  
    const params = [];
    if (stock) params.push(`symbol=${encodeURIComponent(stock)}`);
    if (startDate) params.push(`start=${encodeURIComponent(startDate)}`);
    if (endDate) params.push(`end=${encodeURIComponent(endDate)}`);
    if (params.length) url += '?' + params.join('&');
    
    console.log(`Fetching EDA data from: ${url}`);
    
    fetch(url)
      .then(res => {
        console.log('Response status:', res.status);
        console.log('Response headers:', Object.fromEntries([...res.headers.entries()]));
        return res.text(); // Get raw text first to inspect
      })
      .then(text => {
        console.log('Raw response text:', text.substring(0, 500));
        
        try {
          // Try to parse as JSON
          const data = JSON.parse(text);
          console.log('API data received:', data);
          
          // Set original plots
          setTrendlinesFig(data.trendlines || null);
          setEcdfFig(data.ecdf || null);
          setHistFig(data.hist || null);
          
          // Set new visualizations
          setCandlestickFig(data.candlestick || null);
          
          // Price data visualizations
          setReturnsDistFig(data.returns_dist || null);
          setPriceRangeBoxFig(data.price_range_box || null);
          setPriceRangeViolinFig(data.price_range_violin || null);
          setGapAnalysisBoxFig(data.gap_analysis_box || null);
          setGapAnalysisViolinFig(data.gap_analysis_violin || null);
          setClosePositioningBoxFig(data.close_positioning_box || null);
          setClosePositioningViolinFig(data.close_positioning_violin || null);
          setMultitimeframeBoxFig(data.multitimeframe_box || null);
          setMultitimeframeViolinFig(data.multitimeframe_violin || null);
          setSeasonalBoxFig(data.seasonal_box || null);
          setSeasonalViolinFig(data.seasonal_violin || null);
          
          // Volume data visualizations
          setVolumeDistBoxFig(data.volume_dist_box || null);
          setVolumeDistViolinFig(data.volume_dist_violin || null);
          setRelativeVolumeBoxFig(data.relative_volume_box || null);
          setRelativeVolumeViolinFig(data.relative_volume_violin || null);
          setPriceVolumeBoxFig(data.price_volume_box || null);
          setPriceVolumeViolinFig(data.price_volume_violin || null);
          setVolumePersistenceBoxFig(data.volume_persistence_box || null);
          setVolumePersistenceViolinFig(data.volume_persistence_violin || null);
          setVolumeEventBoxFig(data.volume_event_box || null);
          setVolumeEventViolinFig(data.volume_event_violin || null);
          setVolumePriceLevelBoxFig(data.volume_price_level_box || null);
          setVolumePriceLevelViolinFig(data.volume_price_level_violin || null);
        } catch (e) {
          console.error('JSON parse error:', e);
          throw new Error(`Failed to parse response as JSON. Raw response starts with: ${text.substring(0, 100)}...`);
        }
        
        setLoading(false);
      })
      .catch(err => {
        console.error('Fetch error:', err);
        setError(err.message);
        setLoading(false);
      });
  }, [stock, startDate, endDate]);

  useEffect(() => {
    if (startDate && endDate) {
      fetchEDAData();
    }
  }, [fetchEDAData, startDate, endDate]);

  // Helper function to render a plot if it exists
  const renderPlot = (fig: any, title: string, description: string = '') => {
    if (!fig) return null;

    let layout = fig?.layout || { title };
    const data = fig?.data || [];

    // Special handling for Candlestick chart to separate volume
    if (title === "Candlestick Chart" && data.length > 1 && data[0]?.type === 'candlestick' && data[1]?.type === 'bar') {
      // Assign volume to secondary y-axis
      data[1].yaxis = 'y2'; 

      // Determine volume bar colors based on price movement
      const colors = data[0].close.map((close: number, i: number) => {
        if (close > data[0].open[i]) {
          return '#2CA02C'; // Green for up days
        } else {
          return '#D62728'; // Red for down days (or flat)
        }
      });
      
      // Apply colors to the volume trace (assuming data[1] is volume)
      data[1].marker = { ...(data[1].marker || {}), color: colors };

      layout = {
        ...layout,
        yaxis: { domain: [0.25, 1] }, // Price axis takes top 75%
        yaxis2: { domain: [0, 0.2] }, // Volume axis takes bottom 20%
        xaxis: { 
          rangeslider: { visible: false } // Hide range slider often included by default
        },
        showlegend: false // Often cleaner without legend for this type
      };
    }

    return (
      <div style={{marginTop: 40}}>
        <Typography variant="h6" gutterBottom>{title}</Typography>
        {description && (
          <Typography variant="body2" gutterBottom>{description}</Typography>
        )}
        <Plot 
          data={data} 
          layout={layout} 
          style={{width: '100%', height: '600px'}} 
        />
      </div>
    );
  };

  // Helper function to render paired box and violin plots
  const renderBoxViolinPair = (boxFig: any, violinFig: any, title: string, description: string = '') => {
    if (!boxFig && !violinFig) return null;
    return (
      <div style={{marginTop: 40}}>
        <Typography variant="h6" gutterBottom>{title}</Typography>
        {description && (
          <Typography variant="body2" gutterBottom>{description}</Typography>
        )}
        <Grid container spacing={2}>
          <Grid item xs={12} md={6}>
            {boxFig && (
              <Plot 
                data={boxFig?.data || []} 
                layout={boxFig?.layout || {title: `${title} (Box Plot)`}} 
                style={{width: '100%', height: '600px'}} 
              />
            )}
          </Grid>
          <Grid item xs={12} md={6}>
            {violinFig && (
              <Plot 
                data={violinFig?.data || []} 
                layout={violinFig?.layout || {title: `${title} (Violin Plot)`}} 
                style={{width: '100%', height: '600px'}} 
              />
            )}
          </Grid>
        </Grid>
      </div>
    );
  };

  return (
    <Paper sx={{ /* Removed p: 3 */ }}>
      <Typography variant="h4" gutterBottom>
        Exploratory Data Analysis (EDA)
      </Typography>
      <div style={{ margin: '20px 0' }}>
        <StockAutocomplete />
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
        {loading && <CircularProgress />}
        {error && <Alert severity="error">{error}</Alert>}
      </div>

      {!loading && !error && (
        <>
          <Box sx={{ borderBottom: 1, borderColor: 'divider', mt: 4 }}>
            <Tabs value={tabValue} onChange={handleTabChange} aria-label="EDA tabs">
              <Tab label="Overview" />
              <Tab label="Price Analysis" />
              <Tab label="Volume Analysis" />
            </Tabs>
          </Box>
          
          {/* Overview Tab */}
          <TabPanel value={tabValue} index={0}>
            {renderPlot(
              candlestickFig,
              "Candlestick Chart",
              "OHLC price visualization for the selected time period."
            )}
            
            {renderPlot(
              trendlinesFig,
              "Linear and Non-Linear Trendlines",
              "Apply directly to raw price data to identify overall market direction. Help determine appropriate detrending methods before feature engineering."
            )}
            
            {renderPlot(
              ecdfFig,
              "Empirical Cumulative Distribution Plots (ECDF)",
              "Excellent for raw returns (calculated from close prices). Helps identify the probability of returns falling below certain thresholds."
            )}
            
            {renderPlot(
              histFig,
              "Histograms & Distplots",
              "Use on raw daily returns to check for normality/non-normality. Apply to volume data to understand trading activity distribution."
            )}
          </TabPanel>
          
          {/* Price Analysis Tab */}
          <TabPanel value={tabValue} index={1}>
            {renderPlot(
              returnsDistFig,
              "Distribution of Daily Returns",
              "Daily percentage returns distribution analysis to identify patterns and market regimes."
            )}
            
            {renderBoxViolinPair(
              priceRangeBoxFig,
              priceRangeViolinFig,
              "Price Range Analysis",
              "Box and violin plots of daily trading ranges (High-Low) to understand volatility patterns."
            )}
            
            {renderBoxViolinPair(
              gapAnalysisBoxFig,
              gapAnalysisViolinFig,
              "Gap Analysis",
              "Analysis of overnight gaps (today's open vs previous day's close) to identify patterns in overnight price movements."
            )}
            
            {renderBoxViolinPair(
              closePositioningBoxFig,
              closePositioningViolinFig,
              "Close Relative to Range Positioning",
              "Shows where the closing price typically falls within the daily range, calculated as (Close-Low)/(High-Low)."
            )}
            
            {renderBoxViolinPair(
              multitimeframeBoxFig,
              multitimeframeViolinFig,
              "Multi-timeframe Comparisons",
              "Compares daily price movements against weekly ranges to understand timeframe relationships."
            )}
            
            {renderBoxViolinPair(
              seasonalBoxFig,
              seasonalViolinFig,
              "Seasonal Patterns",
              "Returns grouped by month, day of week, or time of year to identify recurring seasonal patterns."
            )}
          </TabPanel>
          
          {/* Volume Analysis Tab */}
          <TabPanel value={tabValue} index={2}>
            {renderBoxViolinPair(
              volumeDistBoxFig,
              volumeDistViolinFig,
              "Volume Distribution",
              "Basic volume distribution analysis across different timeframes."
            )}
            
            {renderBoxViolinPair(
              relativeVolumeBoxFig,
              relativeVolumeViolinFig,
              "Relative Volume Analysis",
              "Volume relative to moving averages (e.g., Volume/20-day Average Volume) to identify unusual activity."
            )}
            
            {renderBoxViolinPair(
              priceVolumeBoxFig,
              priceVolumeViolinFig,
              "Price-conditioned Volume",
              "Comparison of volume on up days versus down days to identify volume-price relationships."
            )}
            
            {renderBoxViolinPair(
              volumePersistenceBoxFig,
              volumePersistenceViolinFig,
              "Volume Persistence",
              "Analysis of how high/low volume days cluster together over time."
            )}
            
            {renderBoxViolinPair(
              volumeEventBoxFig,
              volumeEventViolinFig,
              "Pre/post Event Volume",
              "Comparing normal volume to volume around significant market or company events."
            )}
            
            {renderBoxViolinPair(
              volumePriceLevelBoxFig,
              volumePriceLevelViolinFig,
              "Volume by Price Level",
              "Analysis of volume at different price levels or during breakout/breakdown scenarios."
            )}
          </TabPanel>
        </>
      )}
    </Paper>
  );
};

export default EDA;
