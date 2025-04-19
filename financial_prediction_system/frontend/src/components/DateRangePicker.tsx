import React from 'react';
import { TextField, Box } from '@mui/material';

interface DateRangePickerProps {
  startDate: string;
  endDate: string;
  minDate: string;
  maxDate: string;
  onChange: (start: string, end: string) => void;
}

const DateRangePicker: React.FC<DateRangePickerProps> = ({ startDate, endDate, minDate, maxDate, onChange }) => {
  return (
    <Box display="flex" alignItems="center" gap={2}>
      <TextField
        label="Start Date"
        type="date"
        size="small"
        InputLabelProps={{ shrink: true }}
        inputProps={{ min: minDate, max: maxDate }}
        value={startDate}
        onChange={e => onChange(e.target.value, endDate)}
      />
      <TextField
        label="End Date"
        type="date"
        size="small"
        InputLabelProps={{ shrink: true }}
        inputProps={{ min: minDate, max: maxDate }}
        value={endDate}
        onChange={e => onChange(startDate, e.target.value)}
      />
    </Box>
  );
};

export default DateRangePicker;
