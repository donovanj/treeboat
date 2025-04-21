import React, { useState, useEffect } from 'react';
import {
    Paper,
    Typography,
    Grid,
    FormControl,
    InputLabel,
    Select,
    MenuItem,
    TextField,
    Button,
    Box,
    Alert,
    Slider,
    FormControlLabel,
    Switch
} from '@mui/material';
import { Line } from 'react-chartjs-2';

interface TargetEngineeringProps {
    baseData: {
        dates: string[];
        prices: number[];
    };
    onTargetChange: (target: number[]) => void;
}

interface TargetConfig {
    type: 'classification' | 'regression' | 'probabilistic' | 'ranking';
    horizon: number;
    threshold?: number;
    volAdjusted: boolean;
    nBins?: number;
}

const defaultConfig: TargetConfig = {
    type: 'classification',
    horizon: 5,
    threshold: 0,
    volAdjusted: false,
    nBins: 5
};

const TargetEngineering: React.FC<TargetEngineeringProps> = ({
    baseData,
    onTargetChange
}) => {
    const [config, setConfig] = useState<TargetConfig>(defaultConfig);
    const [targetData, setTargetData] = useState<number[]>([]);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [stats, setStats] = useState<any>(null);

    const updateTarget = async () => {
        setLoading(true);
        setError(null);

        try {
            const response = await fetch('/api/features/target-preview', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    data: baseData,
                    target_type: config.type,
                    parameters: {
                        horizon: config.horizon,
                        threshold: config.threshold,
                        vol_adjust: config.volAdjusted,
                        n_bins: config.nBins
                    }
                }),
            });

            if (!response.ok) {
                throw new Error('Failed to generate target');
            }

            const data = await response.json();
            setTargetData(data.target);
            setStats(data.metrics);
            onTargetChange(data.target);
        } catch (err: any) {
            setError(err.message);
        } finally {
            setLoading(false);
        }
    };

    useEffect(() => {
        if (baseData.dates.length > 0) {
            updateTarget();
        }
    }, [baseData, config]);

    const chartData = {
        labels: baseData.dates,
        datasets: [
            {
                label: 'Price',
                data: baseData.prices,
                borderColor: 'rgb(75, 192, 192)',
                yAxisID: 'y',
            },
            {
                label: 'Target',
                data: targetData,
                borderColor: 'rgb(255, 99, 132)',
                yAxisID: 'y1',
            },
        ],
    };

    const chartOptions = {
        responsive: true,
        interaction: {
            mode: 'index' as const,
            intersect: false,
        },
        scales: {
            y: {
                type: 'linear' as const,
                display: true,
                position: 'left' as const,
            },
            y1: {
                type: 'linear' as const,
                display: true,
                position: 'right' as const,
                grid: {
                    drawOnChartArea: false,
                },
            },
        },
    };

    return (
        <Paper sx={{ p: 2 }}>
            <Typography variant="h6" gutterBottom>
                Target Variable Engineering
            </Typography>

            <Grid container spacing={3}>
                <Grid item xs={12} md={6}>
                    <FormControl fullWidth>
                        <InputLabel>Target Type</InputLabel>
                        <Select
                            value={config.type}
                            label="Target Type"
                            onChange={(e) => setConfig({
                                ...config,
                                type: e.target.value as TargetConfig['type']
                            })}
                        >
                            <MenuItem value="classification">Classification (Up/Down)</MenuItem>
                            <MenuItem value="regression">Regression (Returns)</MenuItem>
                            <MenuItem value="probabilistic">Probabilistic (Distribution)</MenuItem>
                            <MenuItem value="ranking">Ranking (Relative)</MenuItem>
                        </Select>
                    </FormControl>
                </Grid>

                <Grid item xs={12} md={6}>
                    <Typography gutterBottom>
                        Forecast Horizon (Days)
                    </Typography>
                    <Slider
                        value={config.horizon}
                        min={1}
                        max={60}
                        step={1}
                        marks={[
                            { value: 1, label: '1d' },
                            { value: 5, label: '1w' },
                            { value: 20, label: '1m' },
                            { value: 60, label: '3m' }
                        ]}
                        onChange={(_, value) => setConfig({
                            ...config,
                            horizon: value as number
                        })}
                    />
                </Grid>

                {config.type === 'classification' && (
                    <Grid item xs={12} md={6}>
                        <TextField
                            label="Return Threshold (%)"
                            type="number"
                            value={config.threshold}
                            onChange={(e) => setConfig({
                                ...config,
                                threshold: parseFloat(e.target.value)
                            })}
                            fullWidth
                        />
                    </Grid>
                )}

                {config.type === 'probabilistic' && (
                    <Grid item xs={12} md={6}>
                        <TextField
                            label="Number of Bins"
                            type="number"
                            value={config.nBins}
                            onChange={(e) => setConfig({
                                ...config,
                                nBins: parseInt(e.target.value)
                            })}
                            fullWidth
                        />
                    </Grid>
                )}

                <Grid item xs={12}>
                    <FormControlLabel
                        control={
                            <Switch
                                checked={config.volAdjusted}
                                onChange={(e) => setConfig({
                                    ...config,
                                    volAdjusted: e.target.checked
                                })}
                            />
                        }
                        label="Volatility Adjusted"
                    />
                </Grid>
            </Grid>

            {error && (
                <Alert severity="error" sx={{ mt: 2 }}>
                    {error}
                </Alert>
            )}

            <Box sx={{ mt: 3, height: 300 }}>
                <Line options={chartOptions} data={chartData} />
            </Box>

            {stats && (
                <Box sx={{ mt: 3 }}>
                    <Typography variant="subtitle1" gutterBottom>
                        Target Statistics
                    </Typography>
                    <Grid container spacing={2}>
                        <Grid item xs={6} md={3}>
                            <Typography variant="body2" color="textSecondary">
                                Mean
                            </Typography>
                            <Typography variant="body1">
                                {stats.mean.toFixed(4)}
                            </Typography>
                        </Grid>
                        <Grid item xs={6} md={3}>
                            <Typography variant="body2" color="textSecondary">
                                Std Dev
                            </Typography>
                            <Typography variant="body1">
                                {stats.std.toFixed(4)}
                            </Typography>
                        </Grid>
                        <Grid item xs={6} md={3}>
                            <Typography variant="body2" color="textSecondary">
                                Min
                            </Typography>
                            <Typography variant="body1">
                                {stats.min.toFixed(4)}
                            </Typography>
                        </Grid>
                        <Grid item xs={6} md={3}>
                            <Typography variant="body2" color="textSecondary">
                                Max
                            </Typography>
                            <Typography variant="body1">
                                {stats.max.toFixed(4)}
                            </Typography>
                        </Grid>
                    </Grid>
                </Box>
            )}
        </Paper>
    );
};

export default TargetEngineering;