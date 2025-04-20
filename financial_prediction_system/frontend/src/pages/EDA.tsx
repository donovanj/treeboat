import React, { useState, useEffect, useCallback } from 'react';
import { useGlobalSelection } from '../context/GlobalSelectionContext';
import { Typography, Paper, CircularProgress, Alert, Grid, Tabs, Tab, Box, Button, SxProps, Theme } from '@mui/material';
import Plot from 'react-plotly.js';
import StockAutocomplete from '../components/StockAutocomplete';

// --- Mantine Imports ---
import { Calendar, CalendarProps, DayProps } from '@mantine/dates';
import dayjs from 'dayjs';
import isBetween from 'dayjs/plugin/isBetween';
dayjs.extend(isBetween);
import '@mantine/dates/styles.css';
// ---------------------

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

  // New Gap-Volume visualizations
  const [gapVolumeScatterFig, setGapVolumeScatterFig] = useState<any>(null);
  const [gapVolumeBubbleFig, setGapVolumeBubbleFig] = useState<any>(null);
  const [gapVolumeHeatmapFig, setGapVolumeHeatmapFig] = useState<any>(null);
  const [gapFollowThroughFig, setGapFollowThroughFig] = useState<any>(null);
  const [gapCategoryBoxFig, setGapCategoryBoxFig] = useState<any>(null);
  const [gapVolumeTimeseriesFig, setGapVolumeTimeseriesFig] = useState<any>(null);
  const [volumeWeightedGapsFig, setVolumeWeightedGapsFig] = useState<any>(null);
  const [gapVolumeSurfaceFig, setGapVolumeSurfaceFig] = useState<any>(null);
  const [gapViolinStripFig, setGapViolinStripFig] = useState<any>(null);

  // Volatility Analysis Plots
  const [rollingVolatilityFig, setRollingVolatilityFig] = useState<any>(null);
  const [parkinsonVolatilityFig, setParkinsonVolatilityFig] = useState<any>(null);
  const [atrFig, setAtrFig] = useState<any>(null);
  const [volatilityComparisonFig, setVolatilityComparisonFig] = useState<any>(null);

  // --- Correlation Analysis Plots --- 
  const [rollingCorrelationFig, setRollingCorrelationFig] = useState<any>(null);
  const [correlationMatrixFig, setCorrelationMatrixFig] = useState<any>(null);
  // ------------------------------------

  // --- New Volatility Plots --- 
  const [volatilityTermStructureFig, setVolatilityTermStructureFig] = useState<any>(null);
  const [volatilityConeFig, setVolatilityConeFig] = useState<any>(null);
  const [volatilityRegimeFig, setVolatilityRegimeFig] = useState<any>(null);
  // ---------------------------

  // --- Anomaly Detection Plots ---
  const [zScoreFig, setZScoreFig] = useState<any>(null);
  const [zAnomaliesFig, setZAnomaliesFig] = useState<any>(null);
  const [iforestFig, setIforestFig] = useState<any>(null);
  const [dbscanFig, setDbscanFig] = useState<any>(null);
  const [stlFig, setStlFig] = useState<any>(null);
  const [logReturnsZScoreFig, setLogReturnsZScoreFig] = useState<any>(null);
  // -------------------------------

  // --- Spectral Analysis Plots ---
  const [fftFig, setFftFig] = useState<any>(null);
  const [dwtFig, setDwtFig] = useState<any>(null);
  const [cwtFig, setCwtFig] = useState<any>(null);
  // -------------------------------

  // --- Tail Risk Analysis Plots ---
  const [varComparisonPlot, setVarComparisonPlot] = useState<any>(null);
  const [rollingVarCvarPlot, setRollingVarCvarPlot] = useState<any>(null);
  const [evtReturnLevelPlot, setEvtReturnLevelPlot] = useState<any>(null);
  const [hillPlot, setHillPlot] = useState<any>(null);
  // --------------------------------

  // --- Momentum / Mean Reversion Plots --- 
  const [rsiPlot, setRsiPlot] = useState<any>(null);
  const [macdPlot, setMacdPlot] = useState<any>(null);
  const [bollingerBandsPlot, setBollingerBandsPlot] = useState<any>(null);

  // Local state to manage the date range selection process
  const [selectedRange, setSelectedRange] = useState<[Date | null, Date | null]>([null, null]);

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

  // Effect to initialize dates (both global and local)
  useEffect(() => {
    if (!startDate || !endDate) {
      const { startDate: defaultStart, endDate: defaultEnd } = getDefaultDates();
      // Set global context first
      setStartDate(defaultStart);
      setEndDate(defaultEnd);
      // Initialize local state based on defaults
      setSelectedRange([new Date(defaultStart + 'T00:00:00'), new Date(defaultEnd + 'T00:00:00')]);
    } else {
      // Sync local state if global context already has values
      setSelectedRange([new Date(startDate + 'T00:00:00'), new Date(endDate + 'T00:00:00')]);
    }
    // Intentionally only run once on mount or if global context is initially missing
    // eslint-disable-next-line react-hooks/exhaustive-deps 
  }, []); // Empty dependency array or based on global context presence

  // Effect to sync local state FROM global context changes (e.g., from another component)
  useEffect(() => {
    const globalStart = startDate ? new Date(startDate + 'T00:00:00') : null;
    const globalEnd = endDate ? new Date(endDate + 'T00:00:00') : null;
    const localStart = selectedRange[0];
    const localEnd = selectedRange[1];

    // Check if local state differs from global context and update if needed
    if ( (!localStart && globalStart) || (localStart && globalStart && !dayjs(localStart).isSame(globalStart, 'day')) ||
         (!localEnd && globalEnd) || (localEnd && globalEnd && !dayjs(localEnd).isSame(globalEnd, 'day')) ) {
        console.log("Syncing calendar state from global context change", globalStart, globalEnd);
        setSelectedRange([globalStart, globalEnd]);
    }
  }, [startDate, endDate]); // Depend only on global context changes

  const handleTabChange = (event: React.SyntheticEvent, newValue: number) => {
    setTabValue(newValue);
  };

  // Fetch data based on GLOBAL context dates
  const fetchEDAData = useCallback(() => {
    if (!stock || !startDate || !endDate) {
        console.log("Fetch skipped: Stock or global dates missing.");
        setError(null); // Clear previous errors if selection is incomplete
        // Optionally clear plots here
        return;
    }
    setLoading(true);
    setError(null);
    // Clear previous plots before fetching new data
    setTrendlinesFig(null);
    setEcdfFig(null);
    setHistFig(null);
    setCandlestickFig(null);
    setReturnsDistFig(null);
    setPriceRangeBoxFig(null);
    setPriceRangeViolinFig(null);
    setGapAnalysisBoxFig(null);
    setGapAnalysisViolinFig(null);
    setClosePositioningBoxFig(null);
    setClosePositioningViolinFig(null);
    setMultitimeframeBoxFig(null);
    setMultitimeframeViolinFig(null);
    setSeasonalBoxFig(null);
    setSeasonalViolinFig(null);
    setVolumeDistBoxFig(null);
    setVolumeDistViolinFig(null);
    setRelativeVolumeBoxFig(null);
    setRelativeVolumeViolinFig(null);
    setPriceVolumeBoxFig(null);
    setPriceVolumeViolinFig(null);
    setVolumePersistenceBoxFig(null);
    setVolumePersistenceViolinFig(null);
    setVolumeEventBoxFig(null);
    setVolumeEventViolinFig(null);
    setVolumePriceLevelBoxFig(null);
    setVolumePriceLevelViolinFig(null);
    setRollingVolatilityFig(null);
    setParkinsonVolatilityFig(null);
    setAtrFig(null);
    setVolatilityComparisonFig(null);
    setVolatilityTermStructureFig(null);
    setVolatilityConeFig(null);
    setVolatilityRegimeFig(null);
    setRollingCorrelationFig(null);
    setCorrelationMatrixFig(null);
    setZScoreFig(null);
    setZAnomaliesFig(null);
    setIforestFig(null);
    setDbscanFig(null);
    setStlFig(null);
    setLogReturnsZScoreFig(null);
    setFftFig(null);
    setDwtFig(null);
    setCwtFig(null);
    setVarComparisonPlot(null);
    setRollingVarCvarPlot(null);
    setEvtReturnLevelPlot(null);
    setHillPlot(null);
    setRsiPlot(null);
    setMacdPlot(null);
    setBollingerBandsPlot(null);

    let url = '/api/eda';  
    const params = [];
    if (stock) params.push(`symbol=${encodeURIComponent(stock)}`);
    // Use GLOBAL context dates for the fetch URL
    params.push(`start=${encodeURIComponent(startDate)}`); 
    params.push(`end=${encodeURIComponent(endDate)}`);
    
    url += '?' + params.join('&');
    
    console.log(`Fetching EDA data from: ${url}`);
    
    fetch(url)
      .then(res => {
        console.log('Response status:', res.status);
        console.log('Response headers:', Object.fromEntries([...res.headers.entries()]));
        if (!res.ok) {
          // Handle non-2xx responses
          return res.json().then(errData => {
             throw new Error(errData.error || errData.info || `HTTP error! status: ${res.status}`);
          });
        }
        return res.text(); // Get raw text first to inspect
      })
      .then(text => {
        console.log('Raw response text:', text.substring(0, 500));
        
        try {
          // Try to parse as JSON
          const data = JSON.parse(text);
          console.log('API data received:', data);

          // Check for info message (e.g., data found but no plots)
          if (data.info) {
              setError(data.info); // Display info message as an error/alert
              setLoading(false);
              return; // Stop processing further
          }
          
          // Set original plots with enhanced tooltips
          // Add percentage change to trendlines figure
          if (data.trendlines) {
            console.log('Original trendlines figure:', data.trendlines);
            const enhancedTrendlines = JSON.parse(JSON.stringify(data.trendlines));
            if (enhancedTrendlines.data && Array.isArray(enhancedTrendlines.data)) {
              // Find the candlestick trace (usually the first one)
              const candlestickTrace = enhancedTrendlines.data.find((trace: any) => trace.type === 'candlestick');
              console.log('Candlestick trace in trendlines:', candlestickTrace);
              if (candlestickTrace) {
                // Calculate percentage change for each point
                const close = candlestickTrace.close;
                const open = candlestickTrace.open;
                if (close && open && Array.isArray(close) && Array.isArray(open)) {
                  const pctChange = close.map((c: number, i: number) => {
                    if (c === null || open[i] === null) return null;
                    return ((c - open[i]) / open[i] * 100).toFixed(2) + '%';
                  });
                  
                  // Enhance the hovertemplate to include percentage change
                  candlestickTrace.hovertemplate = 
                    'Date: %{x}<br>' +
                    'Open: %{open}<br>' +
                    'High: %{high}<br>' +
                    'Low: %{low}<br>' +
                    'Close: %{close}<br>' +
                    'Change: %{customdata}<extra></extra>';
                  
                  // Add the percentage change as custom data
                  candlestickTrace.customdata = pctChange;
                }
              }
            }
            console.log('Enhanced trendlines figure:', enhancedTrendlines);
            setTrendlinesFig(enhancedTrendlines);
          } else {
            setTrendlinesFig(null);
          }
          
          setEcdfFig(data.ecdf || null);
          setHistFig(data.hist || null);
          
          // Set new visualizations with enhanced tooltips
          // Add percentage change to candlestick figure
          if (data.candlestick) {
            console.log('Original candlestick figure:', data.candlestick);
            const enhancedCandlestick = JSON.parse(JSON.stringify(data.candlestick));
            if (enhancedCandlestick.data && Array.isArray(enhancedCandlestick.data)) {
              // Find the candlestick trace
              const candlestickTrace = enhancedCandlestick.data.find((trace: any) => trace.type === 'candlestick');
              console.log('Candlestick trace in candlestick figure:', candlestickTrace);
              if (candlestickTrace) {
                // Calculate percentage change for each point
                const close = candlestickTrace.close;
                const open = candlestickTrace.open;
                if (close && open && Array.isArray(close) && Array.isArray(open)) {
                  const pctChange = close.map((c: number, i: number) => {
                    if (c === null || open[i] === null) return null;
                    return ((c - open[i]) / open[i] * 100).toFixed(2) + '%';
                  });
                  
                  // Enhance the hovertemplate to include percentage change
                  candlestickTrace.hovertemplate = 
                    'Date: %{x}<br>' +
                    'Open: %{open}<br>' +
                    'High: %{high}<br>' +
                    'Low: %{low}<br>' +
                    'Close: %{close}<br>' +
                    'Change: %{customdata}<extra></extra>';
                  
                  // Add the percentage change as custom data
                  candlestickTrace.customdata = pctChange;
                }
              }
            }
            console.log('Enhanced candlestick figure:', enhancedCandlestick);
            setCandlestickFig(enhancedCandlestick);
          } else {
            setCandlestickFig(null);
          }
          
          // Price data visualizations
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
          setPriceVolumeBoxFig(data.price_volume_box || null);
          setPriceVolumeViolinFig(data.price_volume_violin || null);
          
          // New Gap-Volume visualizations
          setGapVolumeScatterFig(data.gap_volume_scatter || null);
          setGapVolumeBubbleFig(data.gap_volume_bubble || null);
          setGapVolumeHeatmapFig(data.gap_volume_heatmap || null);
          setGapFollowThroughFig(data.gap_follow_through || null);
          setGapCategoryBoxFig(data.gap_category_box || null);
          setGapVolumeTimeseriesFig(data.gap_volume_timeseries || null);
          setVolumeWeightedGapsFig(data.volume_weighted_gaps || null);
          setGapVolumeSurfaceFig(data.gap_volume_surface || null);
          setGapViolinStripFig(data.gap_violin_strip || null);
          
          // Set Volatility Plots
          setRollingVolatilityFig(data.rolling_volatility || null);
          setParkinsonVolatilityFig(data.parkinson_volatility || null);
          setAtrFig(data.atr || null);
          setVolatilityComparisonFig(data.volatility_comparison || null);
          setVolatilityTermStructureFig(data.volatility_term_structure || null);
          setVolatilityConeFig(data.volatility_cone || null);
          setVolatilityRegimeFig(data.volatility_regime || null);

          // Set Correlation Plots
          setRollingCorrelationFig(data.rolling_correlation || null);
          setCorrelationMatrixFig(data.correlation_matrix || null);

          // --- Set Anomaly Detection Plots ---
          setZScoreFig(data.z_score_plot || null);
          setZAnomaliesFig(data.log_returns_z_anomalies_plot || null); // Key used in backend
          setIforestFig(data.iforest_pca_plot || null);
          setDbscanFig(data.dbscan_pca_plot || null);
          setStlFig(data.stl_decomposition_plot || null);
          setLogReturnsZScoreFig(data.log_returns_z_score_plot || null);
          // ------------------------------------

          // --- Set Spectral Analysis Plots ---
          setFftFig(data.fft_plot || null);
          setDwtFig(data.dwt_plot || null);
          setCwtFig(data.cwt_plot || null);
          // -----------------------------------

          // --- Set Tail Risk Analysis Plots ---
          setVarComparisonPlot(data.var_comparison_plot || null);
          setRollingVarCvarPlot(data.rolling_var_cvar_plot || null);
          setEvtReturnLevelPlot(data.evt_return_level_plot || null);
          setHillPlot(data.hill_plot || null);
          // -------------------------------------

          // --- Set Momentum / Mean Reversion Plots ---
          setRsiPlot(data.rsi_plot || null);
          setMacdPlot(data.macd_plot || null);
          setBollingerBandsPlot(data.bollinger_bands_plots || null);
          // ---------------------------------------------

        } catch (e) {
          console.error('JSON parse error:', e);
          throw new Error(`Failed to parse response as JSON. Raw response starts with: ${text.substring(0, 100)}...`);
        }
        
        setLoading(false);
      })
      .catch(err => {
        console.error('Fetch error:', err);
        setError(err.message); // Display the error message from the backend or fetch failure
        setLoading(false);
      });
  // Depend on stock and GLOBAL dates to trigger fetch
  }, [stock, startDate, endDate]); 

  // Effect to trigger fetch when dependencies change
  useEffect(() => {
    // Fetch data only if stock and global dates are set
    if (stock && startDate && endDate) {
      fetchEDAData();
    } else {
      // Clear plots if selection is incomplete
      setLoading(false);
      setError(null);
      // Clear all plot states... (consider a helper function for this)
        setTrendlinesFig(null);
        setEcdfFig(null);
        // ... etc for all plots
    }
  }, [stock, startDate, endDate, fetchEDAData]);

  // Helper function to render a plot if it exists
  const renderPlot = (fig: any, title: string, description: string = '') => {
    // Log input for debugging
    if (["VaR Comparison (95%)", "Rolling VaR & CVaR (63d, 95%)", "EVT Return Level Plot (GPD)", "Hill Plot for Tail Index"].includes(title)) {
        console.log(`Rendering plot: ${title}`);
        console.log('Raw fig input:', fig);
    }

    if (!fig) return (
        <div style={{marginTop: 40}}>
             <Typography variant="h6" gutterBottom>{title}</Typography>
             {description && <Typography variant="body2" gutterBottom>{description}</Typography>}
             <Alert severity="info">Plot data not available.</Alert>
        </div>
    );

    let layout = fig?.layout || { title };
    const data = fig?.data || [];

    // Log parsed data/layout for debugging
    if (["VaR Comparison (95%)", "Rolling VaR & CVaR (63d, 95%)", "EVT Return Level Plot (GPD)", "Hill Plot for Tail Index"].includes(title)) {
        console.log(`Parsed data for ${title}:`, JSON.stringify(data)); // Use stringify for better inspection
        console.log(`Parsed layout for ${title}:`, JSON.stringify(layout));
    }

    // Special handling for Candlestick chart to separate volume
    if (title === "Candlestick Chart" && data.length > 1 && data[0]?.type === 'candlestick' && data[1]?.type === 'bar') {
      // Assign volume to secondary y-axis
      data[1].yaxis = 'y2'; 

      // Determine volume bar colors based on price movement
      // Ensure open and close arrays exist and have the same length
      if (data[0]?.open && data[0]?.close && data[0].open.length === data[0].close.length) {
          const colors = data[0].close.map((close: number, i: number) => {
              return close > data[0].open[i] ? '#2CA02C' : '#D62728'; // Green up, Red down/flat
          });
           // Apply colors to the volume trace (assuming data[1] is volume)
           data[1].marker = { ...(data[1].marker || {}), color: colors };
      }

      layout = {
        ...layout,
        yaxis: { domain: [0.25, 1], autorange: true, type: 'log' }, // Price axis takes top 75%, maintain log
        yaxis2: { domain: [0, 0.2], autorange: true, type: 'log' }, // Volume axis bottom 20%, maintain log
        xaxis: { 
          rangeslider: { visible: false } // Hide range slider
        },
        showlegend: false // Cleaner without legend
      };
    }
    
    // General layout adjustments for consistency
    layout = {
        ...layout,
        autosize: true, // Ensure plot tries to fill container
        margin: { l: 50, r: 50, t: 80, b: 50 }, // Adjust margins
        paper_bgcolor: 'rgba(0,0,0,0)', // Transparent background
        plot_bgcolor: 'rgba(0,0,0,0)', // Transparent plot area
        // Add font styling later if needed
    };

    // Log final data/layout for debugging
    if (["VaR Comparison (95%)", "Rolling VaR & CVaR (63d, 95%)", "EVT Return Level Plot (GPD)", "Hill Plot for Tail Index"].includes(title)) {
        console.log(`Final data being passed to <Plot> for ${title}:`, JSON.stringify(data));
        console.log(`Final layout being passed to <Plot> for ${title}:`, JSON.stringify(layout));
    }

    return (
      <div style={{marginTop: 40}}>
        <Typography variant="h6" gutterBottom>{layout?.title?.text || title}</Typography>
        {description && (
          <Typography variant="body2" gutterBottom>{description}</Typography>
        )}
        <Plot 
          data={data} 
          layout={layout} 
          style={{width: '100%', minHeight: '600px'}} // Use minHeight 
          useResizeHandler={true} // Important for responsiveness
          config={{responsive: true}} // Ensure plotly responsiveness config
        />
      </div>
    );
  };

  // Helper function to render paired box and violin plots
  const renderBoxViolinPair = (boxFig: any, violinFig: any, title: string, description: string = '') => {
    const boxTitle = boxFig?.layout?.title || `${title} (Box Plot)`;
    const violinTitle = violinFig?.layout?.title || `${title} (Violin Plot)`;
    
    // Render even if one plot is missing
    if (!boxFig && !violinFig) return (
         <div style={{marginTop: 40}}>
             <Typography variant="h6" gutterBottom>{title}</Typography>
             {description && <Typography variant="body2" gutterBottom>{description}</Typography>}
             <Alert severity="info">Box and Violin plot data not available.</Alert>
        </div>
    );

    return (
      <div style={{marginTop: 40}}>
        <Typography variant="h6" gutterBottom>{title}</Typography>
        {description && (
          <Typography variant="body2" gutterBottom>{description}</Typography>
        )}
        <Grid container spacing={2}>
          <Grid item xs={12} md={6}>
             {renderPlot(boxFig, boxTitle)} 
          </Grid>
          <Grid item xs={12} md={6}>
            {renderPlot(violinFig, violinTitle)}
          </Grid>
        </Grid>
      </div>
    );
  };

  // Function to handle date clicks on the Calendar
  // Updates local state immediately, updates global state only when range is complete
  const handleDateChange = (date: Date) => {
    const [start, end] = selectedRange;
    const clickedDate = dayjs(date).startOf('day'); // Normalize to start of day
    let newStart: Date | null = null;
    let newEnd: Date | null = null;
    let updateGlobal = false;

    if (start && end) { // A full range is already selected, start a new range
      newStart = clickedDate.toDate();
      newEnd = null;
      setSelectedRange([newStart, newEnd]);
      // Do not update global context yet
    } else if (start && !end) { // Start date is selected, determine end date
      const currentStartDate = dayjs(start).startOf('day');
      if (clickedDate.isBefore(currentStartDate)) { // NEW: Clicked before start: SWAP dates
         newStart = clickedDate.toDate(); // The earlier date is the new start
         newEnd = start; // The original start date is the new end
         setSelectedRange([newStart, newEnd]);
         updateGlobal = true; // Mark global context for update
      } else { // Clicked on or after start: set end date as normal
         newStart = start; // Keep existing start
         newEnd = clickedDate.toDate();
         setSelectedRange([newStart, newEnd]);
         updateGlobal = true; // Mark global context for update
      }
    } else { // No start date selected yet, set the start date
      newStart = clickedDate.toDate();
      newEnd = null;
      setSelectedRange([newStart, newEnd]);
      // Do not update global context yet
    }

    // Update global context only if a full range was just completed
    if (updateGlobal && newStart && newEnd) {
        const globalStartDateStr = dayjs(newStart).format('YYYY-MM-DD');
        const globalEndDateStr = dayjs(newEnd).format('YYYY-MM-DD');
        console.log("Updating global context after range selection:", globalStartDateStr, globalEndDateStr);
        setStartDate(globalStartDateStr);
        setEndDate(globalEndDateStr);
    }
  };

  // Props for styling days in the Calendar - uses local selectedRange state
  const getDayProps: CalendarProps['getDayProps'] = (date): Partial<DayProps> => {
    const [start, end] = selectedRange; // Use local state for styling
    const day = dayjs(date).startOf('day');
    const startDay = start ? dayjs(start).startOf('day') : null;
    const endDay = end ? dayjs(end).startOf('day') : null;

    const isSelectedStart = !!(startDay && day.isSame(startDay, 'date'));
    const isSelectedEnd = !!(endDay && day.isSame(endDay, 'date'));
    // Adjust isBetween to be inclusive for the end date if it exists
    const isInRange = !!(startDay && endDay && day.isBetween(startDay, endDay, 'date', '[]')); 
    // Or handle single selection highlight if needed
    // const isInRange = !!(startDay && endDay && day.isAfter(startDay) && day.isBefore(endDay));

    const selected = isSelectedStart || isSelectedEnd || (isInRange && !(isSelectedStart || isSelectedEnd)); // Highlight range but not ends

    return {
      // selected: selected ? true : undefined, // Mantine doesn't use 'selected' prop directly here for styling
      style: (theme) => ({
        backgroundColor: isSelectedStart || isSelectedEnd 
            ? theme.colors.blue[6] 
            : selected // Use the calculated 'selected' for in-range background
            ? 'rgba(34, 139, 230, 0.15)' 
            : undefined,
        color: isSelectedStart || isSelectedEnd ? theme.white : selected ? theme.black : undefined,
        // Simpler border radius handling might be needed depending on desired look
        borderRadius: isSelectedStart && !end ? '50%' : 
                      isSelectedStart && end ? '50% 0 0 50%' : 
                      isSelectedEnd ? '0 50% 50% 0' : 
                      selected ? 0 : '50%', 
      }),
      onClick: (event) => { 
        event.preventDefault(); 
        handleDateChange(date);
      },
    };
  };

  return (
    <Paper sx={{
      p: 2, 
      margin: 2, 
      marginLeft: 0, 
      paddingLeft: 0, 
      minHeight: '80vh', 
    }}>
      <Typography variant="h4" gutterBottom>
        Exploratory Data Analysis (EDA)
      </Typography>
      <Grid container spacing={2} alignItems="flex-start" sx={{ mb: 3 }}>
        <Grid item xs={12} sm={6} md={4}>
            <StockAutocomplete />
        </Grid>
         <Grid item xs={12} sm={6} md={8}>
            {/* Use basic Calendar with getDayProps for range styling */}
            <Calendar
              // value={selectedRange[0]} // Don't control single value
              getDayProps={getDayProps} // Apply styling and click handler
              minDate={new Date("2020-01-02")}
              maxDate={new Date()}
              numberOfColumns={2} // Display two months
              highlightToday={true} // Highlight today's date
            />
        </Grid>
      </Grid>
      
        {/* Loading and error states */}
        {loading && <Box sx={{ display: 'flex', justifyContent: 'center', my: 4 }}><CircularProgress /></Box>}
        {error && <Alert severity={error.startsWith("Data found") ? "info" : "error"} sx={{ my: 2 }}>{error}</Alert>}

      {/* Render Tabs only if data is ready (not loading and no critical error) */}
      {!loading && (!error || error.startsWith("Data found")) && (
        <>
          <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
            <Tabs value={tabValue} onChange={handleTabChange} aria-label="EDA tabs" variant="scrollable" scrollButtons="auto">
              <Tab label="Overview" />
              <Tab label="Price Analysis" />
              <Tab label="Volume Analysis" />
              <Tab label="Volatility Analysis" />
              <Tab label="Correlation Analysis" />
              <Tab label="Anomaly Detection" />
              <Tab label="Spectral Analysis" />
              <Tab label="Tail Risk Analysis" />
              <Tab label="Momentum/Mean Reversion" />
            </Tabs>
          </Box>
          
          {/* Overview Tab */}
          <TabPanel value={tabValue} index={0}>
            {renderPlot(
              candlestickFig,
              "Candlestick Chart",
              "OHLC price visualization for the selected time period, with volume overlay."
            )}
            {renderPlot(
              trendlinesFig,
              "Linear and Non-Linear Trendlines",
              "Overall market direction identification using linear and exponential trends on price."
            )}
            {renderPlot(
              ecdfFig,
              "Empirical Cumulative Distribution (ECDF)",
              "Probability distribution of daily returns."
            )}
            {renderPlot(
              histFig,
              "Histogram of Daily Returns",
              "Frequency distribution of daily returns to check for normality."
            )}
          </TabPanel>
          
          {/* Price Analysis Tab */}
          <TabPanel value={tabValue} index={1}>
            {renderBoxViolinPair(
              priceRangeBoxFig,
              priceRangeViolinFig,
              "Daily Price Range (High-Low) by Month",
              "Volatility patterns shown by the daily trading range."
            )}
            {renderBoxViolinPair(
              gapAnalysisBoxFig,
              gapAnalysisViolinFig,
              "Overnight Gap Analysis by Month",
              "Patterns in overnight price movements (Today's Open vs Previous Close)."
            )}
            {renderBoxViolinPair(
              closePositioningBoxFig,
              closePositioningViolinFig,
              "Close Relative to Range by Month",
              "Typical closing price position within the daily range [(C-L)/(H-L)]."
            )}
          </TabPanel>
          
          {/* Volume Analysis Tab */}
          <TabPanel value={tabValue} index={2}>
            {renderBoxViolinPair(
              volumeDistBoxFig,
              volumeDistViolinFig,
              "Volume Distribution by Month",
              "Basic analysis of trading volume distribution over time."
            )}
            {renderBoxViolinPair(
              priceVolumeBoxFig,
              priceVolumeViolinFig,
              "Volume on Up vs Down Days",
              "Comparison of volume during positive and negative return days."
            )}
            {renderPlot(
              gapVolumeScatterFig,
              "Gap vs Volume Scatter",
              "Relationship between overnight gap sizes and daily volume, color-coded by up/down days."
            )}
            {renderPlot(
              gapVolumeBubbleFig,
              "Gap Events Over Time",
              "Bubble chart showing gap events with size representing volume."
            )}
            {renderPlot(
              gapVolumeHeatmapFig,
              "Gap-Volume Distribution",
              "Heatmap showing the frequency distribution of gap size vs volume levels."
            )}
            {renderPlot(
              gapFollowThroughFig,
              "Gap Follow-through Analysis",
              "Analysis of gap size vs subsequent intraday movement, with volume-weighted markers."
            )}
            {renderPlot(
              gapCategoryBoxFig,
              "Volume by Gap Category",
              "Distribution of volume across different gap size categories."
            )}
            {renderPlot(
              gapVolumeTimeseriesFig,
              "Volume and Gaps Over Time",
              "Dual-axis time series showing volume and overnight gaps."
            )}
            {renderPlot(
              volumeWeightedGapsFig,
              "Volume-Weighted Gap Distribution",
              "Gap distribution with bar widths weighted by average volume."
            )}
            {renderPlot(
              gapVolumeSurfaceFig,
              "Gap-Volume-Frequency Surface",
              "3D surface plot showing relationships between gaps, volume, and frequency."
            )}
            {renderPlot(
              gapViolinStripFig,
              "Volume Distribution by Gap Category",
              "Detailed volume distribution for each gap category with individual points shown."
            )}
          </TabPanel>
          
          {/* Volatility Analysis Tab */}
          <TabPanel value={tabValue} index={3}>
             {renderPlot(
              rollingVolatilityFig,
              "Rolling Volatility",
              "Annualized standard deviation of log returns over 5, 21, and 63-day windows."
            )}
            {renderPlot(
              parkinsonVolatilityFig,
              "Parkinson Volatility (21d)",
              "Annualized volatility estimated using the daily high-low range."
            )}
            {renderPlot(
              atrFig,
              "Normalized ATR (14d %)",
              "Average True Range (ATR) normalized by closing price, shown below the price chart."
            )}
            {renderPlot(
              volatilityComparisonFig,
              "Price & Volatility Comparison (5d vs 21d)",
              "Candlestick price chart with short-term (5d) and medium-term (21d) rolling volatility below."
            )}
            {renderPlot(
              volatilityTermStructureFig,
              "Volatility Term Structure Heatmap",
              "Annualized volatility levels across different rolling periods over time."
            )}
            {renderPlot(
              volatilityConeFig,
              "Volatility Cone",
              "Current volatility term structure compared against historical distributions."
            )}
            {renderPlot(
              volatilityRegimeFig,
              "Volatility Regime Classification",
              "Historical periods classified by volatility levels (Low, Normal, High, Extreme) based on 21d rolling volatility."
            )}
          </TabPanel>

          {/* Correlation Analysis Tab */}
          <TabPanel value={tabValue} index={4}>
             {renderPlot(
              rollingCorrelationFig,
              "Rolling Correlations (63d)",
              "Rolling correlation between the stock and other major assets/indices."
            )}
            {renderPlot(
              correlationMatrixFig,
              "Correlation Matrix Heatmap",
              "Static correlation matrix between daily log returns over the selected period."
            )}
          </TabPanel>

          {/* --- Anomaly Detection Tab --- */}
          <TabPanel value={tabValue} index={5}>
            <Typography variant="h4" gutterBottom>Anomaly Detection</Typography>
            {renderPlot(logReturnsZScoreFig, "Z-Scores (Log Returns)", "Z-score and Modified Z-score for log returns. Useful for spotting unusual return magnitudes.")}
            {renderPlot(zAnomaliesFig, "Log Returns with Z-Score Anomalies", "Highlights points in the log return series flagged as anomalous by Z-scores or Modified Z-scores.")}
            {renderPlot(iforestFig, "Isolation Forest Anomalies (PCA)", "PCA visualization of features used for Isolation Forest, highlighting detected anomalies. Good for multivariate outlier detection.")}
            {renderPlot(dbscanFig, "DBSCAN Anomalies (PCA)", "PCA visualization of features used for DBSCAN, highlighting detected noise points (anomalies). Effective for density-based clustering anomalies.")}
            {renderPlot(stlFig, "STL Decomposition", "Seasonal-Trend decomposition using Loess for the closing price. Anomalies might appear as large spikes in the residual component.")}
          </TabPanel>
          {/* ----------------------------- */}
          
          {/* --- Spectral Analysis Tab --- */}
          <TabPanel value={tabValue} index={6}>
             {renderPlot(
                fftFig,
                "FFT Power Spectrum (Log Returns)",
                "Identifies dominant cyclical frequencies in log returns."
            )}
             {renderPlot(
                dwtFig,
                "Discrete Wavelet Transform (DWT)",
                "Multi-resolution analysis showing approximation and detail coefficients."
            )}
             {renderPlot(
                cwtFig,
                "Continuous Wavelet Transform (CWT) Scalogram",
                "Time-frequency representation showing how spectral power changes over time."
            )}
          </TabPanel>
          {/* --------------------------- */}
          
          {/* --- Tail Risk Analysis Tab --- */}
          <TabPanel value={tabValue} index={7}>
              {renderPlot(
                 varComparisonPlot,
                 "VaR Comparison (95%)",
                 "Comparison of Value at Risk calculated using Historical, Parametric (Normal), and Parametric (t-distribution) methods."
             )}
             {renderPlot(
                 rollingVarCvarPlot,
                 "Rolling VaR & CVaR (63d, 95%)",
                 "Historical Value at Risk (VaR) and Conditional VaR (CVaR) calculated over a 63-day rolling window."
             )}
             {renderPlot(
                 evtReturnLevelPlot,
                 "EVT Return Level Plot (GPD)",
                 "Expected loss (%) for different return periods (days), estimated using Generalized Pareto Distribution fitted to tail losses."
             )}
             {renderPlot(
                 hillPlot,
                 "Hill Plot for Tail Index",
                 "Hill Plot that measures the tail risk of a stock's returns. In simple terms, it tells you how likely the stock is to experience extreme price movements (crashes or spikes)"
             )}
          </TabPanel>
          {/* ------------------------------ */}
          {/* --- Momentum / Mean Reversion Tab --- */}
          <TabPanel value={tabValue} index={8}>
            {renderPlot(
              rsiPlot,
              "Relative Strength Index (RSI)",
              "Measures the speed and change of price movements (typically 14-period). Oscillates between 0 and 100."
            )}
            {renderPlot(
              macdPlot,
              "Moving Average Convergence Divergence (MACD)",
              "Shows the relationship between two exponential moving averages of price. Includes MACD line, Signal line, and Histogram."
            )}
            {renderPlot(
              bollingerBandsPlot,
              "Bollinger Bands",
              "Volatility bands placed above and below a moving average. Bands widen with increased volatility and narrow with decreased volatility."
            )}
          </TabPanel>
          {/* ----------------------------------- */}
        </>
      )}
    </Paper>
  );
};

export default EDA;
