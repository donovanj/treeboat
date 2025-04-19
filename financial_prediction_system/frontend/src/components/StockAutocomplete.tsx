import React, { useEffect, useState } from 'react';
import Autocomplete from '@mui/material/Autocomplete';
import TextField from '@mui/material/TextField';
import { useGlobalSelection } from '../context/GlobalSelectionContext';

interface StockOption {
  symbol: string;
  company_name: string;
}

interface StockAutocompleteProps {
  // onSelect prop is removed as component now uses context directly
}

const StockAutocomplete: React.FC<StockAutocompleteProps> = () => {
  const { stock: globalStock, setStock: setGlobalStock } = useGlobalSelection();
  const [options, setOptions] = useState<StockOption[]>([]);
  const [loading, setLoading] = useState(false);
  const [componentReady, setComponentReady] = useState(false);

  useEffect(() => {
    let active = true;
    setLoading(true);
    fetch('/api/stock-list')
      .then(res => res.json())
      .then(data => {
        if (active) {
          setOptions(data || []);
          setLoading(false);
          setComponentReady(true);
        }
      })
      .catch(() => {
        if (active) {
          setLoading(false);
          setComponentReady(true);
        }
      });
    return () => { active = false; };
  }, []);

  const selectedOption = componentReady 
    ? options.find(option => option.symbol === globalStock) || null 
    : null;

  return (
    <Autocomplete
      options={options}
      getOptionLabel={(option) => `${option.symbol} - ${option.company_name}`}
      isOptionEqualToValue={(option, value) => option.symbol === value.symbol}
      filterOptions={(opts, state) =>
        opts.filter(
          o =>
            o.symbol.toLowerCase().includes(state.inputValue.toLowerCase()) ||
            o.company_name.toLowerCase().includes(state.inputValue.toLowerCase())
        )
      }
      loading={loading}
      value={selectedOption}
      onChange={(_, newValue) => {
        setGlobalStock(newValue ? newValue.symbol : null);
      }}
      renderInput={(params) => (
        <TextField {...params} label="Select Stock" variant="outlined" />
      )}
      sx={{ width: 400, mb: 2 }}
    />
  );
};

export default StockAutocomplete;
