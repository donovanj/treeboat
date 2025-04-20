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
        }
        // Process data as needed...
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
           <Typography variant="h5" sx={{ mt: 4, mb: 1 }}>Data Cleaning Results Placeholder</Typography>
           {/* Display results here */}
        </>
      )}
    </Paper>
  );
};

export default DataCleaning;
