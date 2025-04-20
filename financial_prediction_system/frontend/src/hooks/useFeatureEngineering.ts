import { useState, useCallback } from 'react';

interface FeatureMetrics {
    mean: number;
    std: number;
    min: number;
    max: number;
    null_count: number;
    price_correlation?: number;
    returns_correlation?: number;
    autocorrelation?: number;
}

interface TargetMetrics {
    correlation: number | null;
    rank_correlation: number | null;
    mutual_info: number | null;
    target_stats: {
        mean: number;
        std: number;
        min: number;
        max: number;
        null_count: number;
    } | null;
}

interface PreviewResponse {
    feature: number[];
    metrics: FeatureMetrics;
}

interface TargetPreviewResponse {
    target: number[];
    metrics: TargetMetrics;
}

export const useFeatureEngineering = () => {
    const [previewData, setPreviewData] = useState<PreviewResponse | null>(null);
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);

    const previewFeature = useCallback(async (formula: string, data: any) => {
        setIsLoading(true);
        setError(null);
        try {
            const response = await fetch('/api/feature-engineering/feature/preview', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    formula,
                    data,
                }),
            });

            if (!response.ok) {
                throw new Error(`Failed to preview feature: ${response.statusText}`);
            }

            const result = await response.json();
            setPreviewData(result);
            return result;
        } catch (err: any) {
            setError(err.message);
            return null;
        } finally {
            setIsLoading(false);
        }
    }, []);

    const previewTarget = useCallback(async (featureValues: number[], data: any, targetType: string, parameters: any) => {
        setIsLoading(true);
        setError(null);
        try {
            const response = await fetch('/api/feature-engineering/feature/target-preview', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    feature_values: featureValues,
                    data,
                    target_type: targetType,
                    parameters,
                }),
            });

            if (!response.ok) {
                throw new Error(`Failed to preview target: ${response.statusText}`);
            }

            const result = await response.json();
            return result as TargetPreviewResponse;
        } catch (err: any) {
            setError(err.message);
            return null;
        } finally {
            setIsLoading(false);
        }
    }, []);

    const saveFeature = useCallback(async (name: string, formula: string, type: string, symbol: string) => {
        setIsLoading(true);
        setError(null);
        try {
            const response = await fetch('/api/feature-engineering/feature/save', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    name,
                    formula,
                    type,
                    symbol,
                }),
            });

            if (!response.ok) {
                throw new Error(`Failed to save feature: ${response.statusText}`);
            }

            const result = await response.json();
            return result;
        } catch (err: any) {
            setError(err.message);
            return null;
        } finally {
            setIsLoading(false);
        }
    }, []);

    return {
        previewFeature,
        saveFeature,
        previewTarget,
        previewData,
        isLoading,
        error,
    };
};

export default useFeatureEngineering;