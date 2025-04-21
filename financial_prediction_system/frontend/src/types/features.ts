export interface FeatureParameter {
    name: string;
    type: string;
    defaultValue?: any;
    description?: string;
}

export interface FeatureConfig {
    type: string;
    parameters: Record<string, any>;
    name?: string;
    description?: string;
}

export interface FeatureComposition {
    id: string;
    name: string;
    symbol: string;
    description?: string;
    features: FeatureConfig[];
    created_at?: string;
    updated_at?: string;
    type?: string;
    parameters?: Record<string, any>;
}

export interface FeatureMetrics {
    mean: number;
    std: number;
    min: number;
    max: number;
    price_correlation?: number;
    returns_correlation?: number;
    mutual_info_score?: number;
}

export interface PreviewResponse {
    features: number[][];
    target?: number[];
    metrics: FeatureMetrics;
}

export interface FeatureImportance {
    feature_name: string;
    correlation: number;
    mutual_info: number;
}

export interface BacktestResult {
    feature_metrics: Record<string, number>;
    stability_metrics: Record<string, number>;
    cv_metrics?: Record<string, number>;
    time_series?: {
        dates: string[];
        target: number[];
        features: number[][];
    };
}

export interface Target {
    id: string;
    name: string;
    type: 'classification' | 'regression' | 'probabilistic' | 'ranking';
    parameters: {
        horizon: number;
        threshold?: number;
        vol_adjust?: boolean;
        n_bins?: number;
        [key: string]: any;
    };
}

export type FeatureType = 
    | 'technical'
    | 'volatility'
    | 'volume'
    | 'market_regime'
    | 'sector'
    | 'yield_curve'
    | 'seasonality'
    | 'custom';

export const FEATURE_TYPE_DESCRIPTIONS: Record<FeatureType, string> = {
    technical: 'Technical analysis indicators like moving averages, RSI, MACD',
    volatility: 'Volatility-based features like standard deviation, ATR',
    volume: 'Volume-based indicators and analysis',
    market_regime: 'Market regime classification and trend analysis',
    sector: 'Sector-relative performance and analysis',
    yield_curve: 'Treasury yield curve features',
    seasonality: 'Time-based seasonal patterns',
    custom: 'Custom user-defined features'
};