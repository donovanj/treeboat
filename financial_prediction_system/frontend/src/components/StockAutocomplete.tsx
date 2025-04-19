import React, { useEffect, useState } from 'react';
import Autocomplete from '@mui/material/Autocomplete';
import TextField from '@mui/material/TextField';

interface StockOption {
  symbol: string;
  company_name: string;
}

interface StockAutocompleteProps {
  onSelect: (symbol: string) => void;
}

const StockAutocomplete: React.FC<StockAutocompleteProps> = ({ onSelect }) => {
  const [options, setOptions] = useState<StockOption[]>([]);
  const [inputValue, setInputValue] = useState('');
  const [value, setValue] = useState<StockOption | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    setLoading(true);
    fetch('/api/stock-list')
      .then(res => res.json())
      .then(data => {
        setOptions(data);
        setLoading(false);
      });
  }, []);

  return (
    <Autocomplete
      options={options}
      getOptionLabel={(option) => `${option.symbol} - ${option.company_name}`}
      filterOptions={(opts, state) =>
        opts.filter(
          o =>
            o.symbol.toLowerCase().includes(state.inputValue.toLowerCase()) ||
            o.company_name.toLowerCase().includes(state.inputValue.toLowerCase())
        )
      }
      loading={loading}
      value={value}
      onChange={(_, newValue) => {
        setValue(newValue);
        if (newValue) onSelect(newValue.symbol);
      }}
      inputValue={inputValue}
      onInputChange={(_, newInputValue) => setInputValue(newInputValue)}
      renderInput={(params) => (
        <TextField {...params} label="Select Stock" variant="outlined" />
      )}
      sx={{ width: 400, mb: 2 }}
    />
  );
};

export default StockAutocomplete;
