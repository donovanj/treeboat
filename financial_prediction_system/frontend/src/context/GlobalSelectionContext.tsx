import React, { createContext, useContext, useState, ReactNode, useEffect } from 'react';

interface GlobalSelectionContextType {
  stock: string | null;
  setStock: (s: string | null) => void;
  startDate: string;
  setStartDate: (d: string) => void;
  endDate: string;
  setEndDate: (d: string) => void;
}

const DEFAULT_END = new Date().toISOString().slice(0, 10);
const DEFAULT_START = new Date(Date.now() - 180 * 24 * 60 * 60 * 1000).toISOString().slice(0, 10); // 6 months ago

const GlobalSelectionContext = createContext<GlobalSelectionContextType | undefined>(undefined);

export const GlobalSelectionProvider = ({ children }: { children: ReactNode }) => {
  // Initialize state from localStorage or use defaults
  const [stock, setStockInternal] = useState<string | null>(() => {
    return localStorage.getItem('selectedStock') || null; // Default to null if not found
  });
  const [startDate, setStartDateInternal] = useState<string>(() => {
    return localStorage.getItem('selectedStartDate') || DEFAULT_START;
  });
  const [endDate, setEndDateInternal] = useState<string>(() => {
    return localStorage.getItem('selectedEndDate') || DEFAULT_END;
  });

  // Wrap setters to update both state and localStorage
  const setStock = (s: string | null) => {
    setStockInternal(s);
    if (s) {
      localStorage.setItem('selectedStock', s);
    } else {
      localStorage.removeItem('selectedStock');
    }
  };

  const setStartDate = (d: string) => {
    setStartDateInternal(d);
    localStorage.setItem('selectedStartDate', d);
  };

  const setEndDate = (d: string) => {
    setEndDateInternal(d);
    localStorage.setItem('selectedEndDate', d);
  };

  return (
    <GlobalSelectionContext.Provider value={{ stock, setStock, startDate, setStartDate, endDate, setEndDate }}>
      {children}
    </GlobalSelectionContext.Provider>
  );
};

export const useGlobalSelection = () => {
  const ctx = useContext(GlobalSelectionContext);
  if (!ctx) throw new Error('useGlobalSelection must be used within GlobalSelectionProvider');
  return ctx;
};
