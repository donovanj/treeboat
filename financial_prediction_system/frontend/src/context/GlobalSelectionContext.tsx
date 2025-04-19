import React, { createContext, useContext, useState, ReactNode } from 'react';

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
  const [stock, setStock] = useState<string | null>(null);
  const [startDate, setStartDate] = useState<string>(DEFAULT_START);
  const [endDate, setEndDate] = useState<string>(DEFAULT_END);

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
