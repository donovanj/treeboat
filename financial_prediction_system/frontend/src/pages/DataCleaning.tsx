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
  TableBody, 
  Box
} from '@mui/material';
import StockAutocomplete from '../components/StockAutocomplete';
import Plot from 'react-plotly.js';

// --- Mantine Imports ---
import { Calendar, CalendarProps, DayProps } from '@mantine/dates';
import dayjs from 'dayjs';
import isBetween from 'dayjs/plugin/isBetween';
dayjs.extend(isBetween);
import '@mantine/dates/styles.css';
// --------------------------

const DataCleaning: React.FC = () => {
  const { stock, setStock, startDate, setStartDate, endDate, setEndDate } = useGlobalSelection();
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [data, setData] = useState<any>(null);
  
  // Local state for range selection
  const [selectedRange, setSelectedRange] = useState<[Date | null, Date | null]>([null, null]);

  // Calculate default dates (keep this)
  const getDefaultDates = useCallback(() => {
    const today = new Date();
    const oneYearAgo = new Date();
    oneYearAgo.setFullYear(today.getFullYear() - 1);
    return {
      startDate: oneYearAgo.toISOString().slice(0, 10),
      endDate: today.toISOString().slice(0, 10)
    };
  }, []);

  // Initialize dates (global and local)
  useEffect(() => {
    if (!startDate || !endDate) {
      const { startDate: defaultStart, endDate: defaultEnd } = getDefaultDates();
      setStartDate(defaultStart);
      setEndDate(defaultEnd);
      setSelectedRange([new Date(defaultStart + 'T00:00:00'), new Date(defaultEnd + 'T00:00:00')]);
    } else {
      setSelectedRange([new Date(startDate + 'T00:00:00'), new Date(endDate + 'T00:00:00')]);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Sync local FROM global
  useEffect(() => {
    const globalStart = startDate ? new Date(startDate + 'T00:00:00') : null;
    const globalEnd = endDate ? new Date(endDate + 'T00:00:00') : null;
    const localStart = selectedRange[0];
    const localEnd = selectedRange[1];

    if ( (!localStart && globalStart) || (localStart && globalStart && !dayjs(localStart).isSame(globalStart, 'day')) ||
         (!localEnd && globalEnd) || (localEnd && globalEnd && !dayjs(localEnd).isSame(globalEnd, 'day')) ) {
      setSelectedRange([globalStart, globalEnd]);
    }
  }, [startDate, endDate]);

  // Fetch data based on GLOBAL dates
  const fetchData = useCallback(() => {
     if (!stock || !startDate || !endDate) {
        console.log("DataCleaning fetch skipped: Stock or global dates missing.");
        setLoading(false); // Ensure loading stops if fetch is skipped
        setError(null);
        return;
    }
    setLoading(true);
    setError(null);
    let url = '/api/data-cleaning';
    const params = [];
    if (stock) params.push(`symbol=${encodeURIComponent(stock)}`);
    // Use GLOBAL dates for API call
    params.push(`start=${encodeURIComponent(startDate)}`);
    params.push(`end=${encodeURIComponent(endDate)}`);
    url += '?' + params.join('&');

    console.log(`DataCleaning: Fetching data from: ${url}`);
    fetch(url)
      .then(res => {
        if (!res.ok) throw new Error('Failed to fetch data');
        return res.json();
      })
      .then(data => {
        console.log('DataCleaning API response:', data); 
        if (!data.success) {
             setError(data.error || 'Unknown error processing data cleaning results');
        } else {
             setData(data);
        }
        setLoading(false);
      })
      .catch(err => {
        setError(err.message);
        setLoading(false);
      });
  // Depend on GLOBAL dates
  }, [stock, startDate, endDate]);

  // Trigger fetch when GLOBAL dates change
  useEffect(() => {
    if (stock && startDate && endDate) {
      fetchData();
    }
  }, [stock, startDate, endDate, fetchData]);

  // Handle date clicks, update local state, update global only when range is complete
  const handleDateChange = (date: Date) => {
    const [start, end] = selectedRange;
    const clickedDate = dayjs(date).startOf('day');
    let newStart: Date | null = null;
    let newEnd: Date | null = null;
    let updateGlobal = false;

    if (start && end) { // A full range is already selected, start a new range
      newStart = clickedDate.toDate();
      newEnd = null;
      setSelectedRange([newStart, newEnd]);
    } else if (start && !end) { // Start date is selected, determine end date
      const currentStartDate = dayjs(start).startOf('day');
      if (clickedDate.isBefore(currentStartDate)) { // NEW: Clicked before start: SWAP dates
         newStart = clickedDate.toDate(); // Earlier date is new start
         newEnd = start; // Original start is new end
         setSelectedRange([newStart, newEnd]);
         updateGlobal = true;
      } else { // Clicked on or after start: set end date as normal
         newStart = start;
         newEnd = clickedDate.toDate();
         setSelectedRange([newStart, newEnd]);
         updateGlobal = true; 
      }
    } else { // No start date selected yet, set the start date
      newStart = clickedDate.toDate();
      newEnd = null;
      setSelectedRange([newStart, newEnd]);
    }

    if (updateGlobal && newStart && newEnd) {
        const globalStartDateStr = dayjs(newStart).format('YYYY-MM-DD');
        const globalEndDateStr = dayjs(newEnd).format('YYYY-MM-DD');
        setStartDate(globalStartDateStr);
        setEndDate(globalEndDateStr);
    }
  };

  // Style days based on LOCAL state
  const getDayProps: CalendarProps['getDayProps'] = (date): Partial<DayProps> => {
    const [start, end] = selectedRange;
    const day = dayjs(date).startOf('day');
    const startDay = start ? dayjs(start).startOf('day') : null;
    const endDay = end ? dayjs(end).startOf('day') : null;

    const isSelectedStart = !!(startDay && day.isSame(startDay, 'date'));
    const isSelectedEnd = !!(endDay && day.isSame(endDay, 'date'));
    const isInRange = !!(startDay && endDay && day.isBetween(startDay, endDay, 'date', '[]'));
    const selected = isSelectedStart || isSelectedEnd || (isInRange && !(isSelectedStart || isSelectedEnd));

    return {
      style: (theme) => ({
        backgroundColor: isSelectedStart || isSelectedEnd 
            ? theme.colors.blue[6] 
            : selected 
            ? 'rgba(34, 139, 230, 0.15)'
            : undefined,
        color: isSelectedStart || isSelectedEnd ? theme.white : selected ? theme.black : undefined,
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
    <Paper sx={{ p: 3 }}>
      <Typography variant="h4" gutterBottom>
        Data Cleaning
      </Typography>
      
      <Grid container spacing={3} sx={{ mb: 3 }} alignItems="flex-start"> 
        <Grid item xs={12} md={6}> 
          <StockAutocomplete />
        </Grid>
        <Grid item xs={12} md={6}>
          {/* Basic Calendar with getDayProps */}
          <Calendar
             getDayProps={getDayProps}
             minDate={new Date("2020-01-02")}
             maxDate={new Date()}
             numberOfColumns={2}
             highlightToday={true}
           />
        </Grid>
      </Grid>
      
      {loading && <Box sx={{ display: 'flex', justifyContent: 'center', my: 4 }}><CircularProgress /></Box>}
      {error && <Alert severity="error" sx={{ my: 2 }}>{error}</Alert>}
      
      {!loading && !error && (
        <>
           <Typography variant="h5" sx={{ mt: 4, mb: 1 }}>Data Cleaning Results</Typography>
           
           {/* Stocks Table Summary */}
           <Paper sx={{ p: 2, mb: 3 }}>
             <Typography variant="h6" gutterBottom>Stocks Table Analysis</Typography>
             <Grid container spacing={2}>
               <Grid item xs={12} md={6}>
                 <TableContainer>
                   <Table size="small">
                     <TableHead>
                       <TableRow>
                         <TableCell>Metric</TableCell>
                         <TableCell>Value</TableCell>
                       </TableRow>
                     </TableHead>
                     <TableBody>
                       {data?.summary && Object.entries(data.summary).map(([key, value]) => (
                         <TableRow key={key}>
                           <TableCell>{key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}</TableCell>
                           <TableCell>{value as string}</TableCell>
                         </TableRow>
                       ))}
                     </TableBody>
                   </Table>
                 </TableContainer>
               </Grid>
               <Grid item xs={12} md={6}>
                 <Typography variant="subtitle1" gutterBottom>Issues Detected</Typography>
                 {data?.issues && Object.entries(data.issues).map(([key, value]) => {
                   const issues = value as string[];
                   return issues.length > 0 ? (
                     <Box key={key} sx={{ mb: 2 }}>
                       <Typography variant="body2" color="error">
                         {key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())} ({issues.length}):
                       </Typography>
                       <ul style={{ margin: 0, paddingLeft: '20px' }}>
                         {issues.slice(0, 5).map((issue, idx) => (
                           <li key={idx}><Typography variant="body2">{issue}</Typography></li>
                         ))}
                         {issues.length > 5 && <li><Typography variant="body2">...and {issues.length - 5} more</Typography></li>}
                       </ul>
                     </Box>
                   ) : null;
                 })}
               </Grid>
             </Grid>
           </Paper>

           {/* Price Data Analysis */}
           <Paper sx={{ p: 2, mb: 3 }}>
             <Typography variant="h6" gutterBottom>Price Data Analysis</Typography>
             
             {/* Price Issues */}
             {data?.price_issues && Object.keys(data.price_issues).length > 0 && (
               <Box sx={{ mb: 3 }}>
                 <Typography variant="subtitle1" color="error" gutterBottom>Issues Detected</Typography>
                 <Grid container spacing={2}>
                   {Object.entries(data.price_issues).map(([key, value]) => {
                     if (key === 'inconsistent_rows' && Array.isArray(value) && value.length > 0) {
                       return (
                         <Grid item xs={12} key={key}>
                           <Typography variant="body2" color="error">
                             {key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())} ({value.length}):
                           </Typography>
                           <TableContainer sx={{ maxHeight: 200 }}>
                             <Table size="small">
                               <TableHead>
                                 <TableRow>
                                   <TableCell>Date</TableCell>
                                   <TableCell>Open</TableCell>
                                   <TableCell>High</TableCell>
                                   <TableCell>Low</TableCell>
                                   <TableCell>Close</TableCell>
                                 </TableRow>
                               </TableHead>
                               <TableBody>
                                 {value.map((row: any, idx: number) => (
                                   <TableRow key={idx}>
                                     <TableCell>{row.date}</TableCell>
                                     <TableCell>{row.open}</TableCell>
                                     <TableCell>{row.high}</TableCell>
                                     <TableCell>{row.low}</TableCell>
                                     <TableCell>{row.close}</TableCell>
                                   </TableRow>
                                 ))}
                               </TableBody>
                             </Table>
                           </TableContainer>
                         </Grid>
                       );
                     } else if (key === 'missing_dates' && Array.isArray(value) && value.length > 0) {
                       return (
                         <Grid item xs={12} md={6} key={key}>
                           <Typography variant="body2" color="error">
                             {key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())} ({value.length}):
                           </Typography>
                           <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5 }}>
                             {value.map((date: string, idx: number) => (
                               <Box key={idx} sx={{ bgcolor: 'error.light', color: 'error.contrastText', px: 1, borderRadius: 1 }}>
                                 {date}
                               </Box>
                             ))}
                             {value.length > 10 && <Box sx={{ px: 1 }}>...and more</Box>}
                           </Box>
                         </Grid>
                       );
                     } else {
                       return (
                         <Grid item xs={12} md={6} key={key}>
                           <Typography variant="body2" color="error">
                             {key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}:
                           </Typography>
                           <Typography variant="body2">{value as string}</Typography>
                         </Grid>
                       );
                     }
                   })}
                 </Grid>
               </Box>
             )}

             {/* Descriptive Statistics */}
             {data?.describe_stats && (
               <Box sx={{ mb: 3 }}>
                 <Typography variant="subtitle1" gutterBottom>Descriptive Statistics</Typography>
                 <Grid container spacing={2}>
                   {Object.entries(data.describe_stats).map(([key, stats]) => (
                     <Grid item xs={12} md={4} key={key}>
                       <Typography variant="body2" fontWeight="bold">{key.toUpperCase()}</Typography>
                       <TableContainer>
                         <Table size="small">
                           <TableBody>
                             {Object.entries(stats as Record<string, number>).map(([stat, value]) => (
                               <TableRow key={stat}>
                                 <TableCell>{stat}</TableCell>
                                 <TableCell>{typeof value === 'number' ? value.toFixed(2) : value}</TableCell>
                               </TableRow>
                             ))}
                           </TableBody>
                         </Table>
                       </TableContainer>
                     </Grid>
                   ))}
                 </Grid>
               </Box>
             )}
           </Paper>

           {/* Visualizations */}
           <Paper sx={{ p: 2, mb: 3 }}>
             <Typography variant="h6" gutterBottom>Visualizations</Typography>
             
             <Grid container spacing={3}>
               {/* OHLCV Chart */}
               {data?.ohlcv_fig && (
                 <Grid item xs={12} lg={6}>
                   <Paper elevation={2} sx={{ p: 1 }}>
                     <Plot
                       data={data.ohlcv_fig.data}
                       layout={data.ohlcv_fig.layout}
                       style={{ width: '100%', height: '400px' }}
                       config={{ responsive: true }}
                     />
                   </Paper>
                 </Grid>
               )}

               {/* Bands Chart */}
               {data?.bands_fig && (
                 <Grid item xs={12} lg={6}>
                   <Paper elevation={2} sx={{ p: 1 }}>
                     <Plot
                       data={data.bands_fig.data}
                       layout={data.bands_fig.layout}
                       style={{ width: '100%', height: '400px' }}
                       config={{ responsive: true }}
                     />
                   </Paper>
                 </Grid>
               )}

               {/* Trendlines Chart */}
               {data?.trendlines_fig && (
                 <Grid item xs={12} lg={6}>
                   <Paper elevation={2} sx={{ p: 1 }}>
                     <Plot
                       data={data.trendlines_fig.data}
                       layout={data.trendlines_fig.layout}
                       style={{ width: '100%', height: '400px' }}
                       config={{ responsive: true }}
                     />
                   </Paper>
                 </Grid>
               )}

               {/* Box/Violin Chart */}
               {data?.box_violin_fig && (
                 <Grid item xs={12} lg={6}>
                   <Paper elevation={2} sx={{ p: 1 }}>
                     <Plot
                       data={data.box_violin_fig.data}
                       layout={data.box_violin_fig.layout}
                       style={{ width: '100%', height: '400px' }}
                       config={{ responsive: true }}
                     />
                   </Paper>
                 </Grid>
               )}

               {/* Missing Data Heatmap */}
               {data?.missing_heatmap_fig && (
                 <Grid item xs={12}>
                   <Paper elevation={2} sx={{ p: 1 }}>
                     <Plot
                       data={data.missing_heatmap_fig.data}
                       layout={data.missing_heatmap_fig.layout}
                       style={{ width: '100%', height: '300px' }}
                       config={{ responsive: true }}
                     />
                   </Paper>
                 </Grid>
               )}

               {/* Legacy Candlestick Chart */}
               {data?.plotly_fig && (
                 <Grid item xs={12}>
                   <Paper elevation={2} sx={{ p: 1 }}>
                     <Plot
                       data={data.plotly_fig.data}
                       layout={data.plotly_fig.layout}
                       style={{ width: '100%', height: '400px' }}
                       config={{ responsive: true }}
                     />
                   </Paper>
                 </Grid>
               )}
             </Grid>
           </Paper>
        </>
      )}
    </Paper>
  );
};

export default DataCleaning;
