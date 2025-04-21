import React, { createContext, useContext, useState, useCallback, ReactNode } from 'react';
import { FeatureComposition } from '../hooks/useFeatureEngineering';

interface StockFeatureContextType {
    selectedFeatures: FeatureComposition[];
    previewData: any;
    isLoading: boolean;
    error: string | null;
    addFeature: (feature: FeatureComposition) => void;
    removeFeature: (id: string) => void;
    updateFeature: (id: string, updates: Partial<FeatureComposition>) => void;
    previewFeatures: () => Promise<void>;
    saveFeatures: () => Promise<void>;
    clearFeatures: () => void;
}

const StockFeatureContext = createContext<StockFeatureContextType | undefined>(undefined);

export const useStockFeature = () => {
    const context = useContext(StockFeatureContext);
    if (!context) {
        throw new Error('useStockFeature must be used within a StockFeatureProvider');
    }
    return context;
};

interface StockFeatureProviderProps {
    children: ReactNode;
}

export const StockFeatureProvider: React.FC<StockFeatureProviderProps> = ({ children }) => {
    const [selectedFeatures, setSelectedFeatures] = useState<FeatureComposition[]>([]);
    const [previewData, setPreviewData] = useState<any>(null);
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);

    const addFeature = useCallback((feature: FeatureComposition) => {
        setSelectedFeatures(prev => [...prev, feature]);
    }, []);

    const removeFeature = useCallback((id: string) => {
        setSelectedFeatures(prev => prev.filter(f => f.id !== id));
    }, []);

    const updateFeature = useCallback((id: string, updates: Partial<FeatureComposition>) => {
        setSelectedFeatures(prev => 
            prev.map(f => f.id === id ? { ...f, ...updates } : f)
        );
    }, []);

    const previewFeatures = useCallback(async () => {
        if (selectedFeatures.length === 0) {
            setError('No features selected');
            return;
        }

        setIsLoading(true);
        setError(null);

        try {
            const response = await fetch('/api/features/preview', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    features: selectedFeatures.map(f => ({
                        type: f.type,
                        parameters: f.parameters
                    }))
                }),
            });

            if (!response.ok) {
                throw new Error('Failed to preview features');
            }

            const data = await response.json();
            setPreviewData(data);
        } catch (err: any) {
            setError(err.message);
        } finally {
            setIsLoading(false);
        }
    }, [selectedFeatures]);

    const saveFeatures = useCallback(async () => {
        if (selectedFeatures.length === 0) {
            setError('No features to save');
            return;
        }

        setIsLoading(true);
        setError(null);

        try {
            const response = await fetch('/api/features/save', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    name: `Feature_Composition_${Date.now()}`,
                    features: selectedFeatures.map(f => ({
                        type: f.type,
                        parameters: f.parameters
                    })),
                    type: 'composition'
                }),
            });

            if (!response.ok) {
                throw new Error('Failed to save features');
            }

            // Clear selected features after successful save
            clearFeatures();
        } catch (err: any) {
            setError(err.message);
        } finally {
            setIsLoading(false);
        }
    }, [selectedFeatures]);

    const clearFeatures = useCallback(() => {
        setSelectedFeatures([]);
        setPreviewData(null);
        setError(null);
    }, []);

    const value = {
        selectedFeatures,
        previewData,
        isLoading,
        error,
        addFeature,
        removeFeature,
        updateFeature,
        previewFeatures,
        saveFeatures,
        clearFeatures
    };

    return (
        <StockFeatureContext.Provider value={value}>
            {children}
        </StockFeatureContext.Provider>
    );
};