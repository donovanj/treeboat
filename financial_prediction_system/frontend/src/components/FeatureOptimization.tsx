import React, { useState, useEffect } from 'react';
import {
    Paper,
    Typography,
    Grid,
    Box,
    Slider,
    TextField,
    Button,
    CircularProgress,
    Alert,
    Table,
    TableBody,
    TableCell,
    TableContainer,
    TableHead,
    TableRow,
    Card,
    CardContent,
    Autocomplete,
    Switch,
    FormControlLabel
} from '@mui/material';
import { FeatureComposition } from '../types/features';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

interface OptimizationResult {
    parameters: Record<string, number | string>;
    metrics: {
        correlation: number;
        mutual_info: number;
        r2: number;
        stability: number;
        prediction_power: number;
    };
    performance_change: number;
}

interface FeatureOptimizationProps {
    composition: FeatureComposition;
    onOptimizationComplete?: (result: OptimizationResult) => void;
    startDate: string;
    endDate: string;
}

const FeatureOptimization: React.FC<FeatureOptimizationProps> = ({
    composition,
    onOptimizationComplete,
    startDate,
    endDate
}) => {
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [optimizationResults, setOptimizationResults] = useState<OptimizationResult[]>([]);
    const [parameters, setParameters] = useState<Record<string, number | string>>({});
    const [autoOptimize, setAutoOptimize] = useState(false);
    const [optimizationProgress, setOptimizationProgress] = useState(0);
    const [parameterRanges, setParameterRanges] = useState<Record<string, [number, number]>>({});
    const [selectedMetric, setSelectedMetric] = useState<string>('prediction_power');

    useEffect(() => {
        detectOptimizableParameters();
    }, [composition]);

    const detectOptimizableParameters = () => {
        if (!composition.formula) return;

        const newParameters: Record<string, number | string> = {};
        const newRanges: Record<string, [number, number]> = {};

        // Detect window sizes
        const windowMatch = composition.formula.match(/rolling\((\d+)\)/g);
        if (windowMatch) {
            const currentWindow = parseInt(windowMatch[0].match(/\d+/)![0]);
            newParameters['window_size'] = currentWindow;
            newRanges['window_size'] = [Math.max(5, currentWindow - 10), currentWindow + 10];
        }

        // Detect thresholds
        const thresholdMatch = composition.formula.match(/threshold[=\s]+(\d+\.?\d*)/g);
        if (thresholdMatch) {
            const currentThreshold = parseFloat(thresholdMatch[0].match(/\d+\.?\d*/)![0]);
            newParameters['threshold'] = currentThreshold;
            newRanges['threshold'] = [currentThreshold * 0.5, currentThreshold * 1.5];
        }

        // Detect alpha/beta parameters
        const alphaMatch = composition.formula.match(/alpha[=\s]+(\d+\.?\d*)/g);
        if (alphaMatch) {
            const currentAlpha = parseFloat(alphaMatch[0].match(/\d+\.?\d*/)![0]);
            newParameters['alpha'] = currentAlpha;
            newRanges['alpha'] = [Math.max(0.01, currentAlpha - 0.1), Math.min(0.99, currentAlpha + 0.1)];
        }

        setParameters(newParameters);
        setParameterRanges(newRanges);
    };

    const optimizeParameters = async () => {
        setLoading(true);
        setError(null);
        const results: OptimizationResult[] = [];

        try {
            // Grid search over parameter combinations
            const paramNames = Object.keys(parameters);
            const totalIterations = autoOptimize ? 20 : 1;
            
            for (let i = 0; i < totalIterations; i++) {
                setOptimizationProgress((i / totalIterations) * 100);
                
                const paramSet = autoOptimize 
                    ? generateRandomParameters()
                    : parameters;

                const response = await fetch('/api/features/optimize', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        feature_id: composition.id,
                        parameters: paramSet,
                        start_date: startDate,
                        end_date: endDate
                    })
                });

                if (!response.ok) {
                    throw new Error('Optimization request failed');
                }

                const result = await response.json();
                results.push({
                    parameters: paramSet,
                    metrics: result.metrics,
                    performance_change: result.performance_change
                });

                // Sort by selected metric
                results.sort((a, b) => 
                    b.metrics[selectedMetric as keyof typeof b.metrics] - 
                    a.metrics[selectedMetric as keyof typeof a.metrics]
                );
            }

            setOptimizationResults(results);
            
            if (results.length > 0 && onOptimizationComplete) {
                onOptimizationComplete(results[0]);
            }

        } catch (err: any) {
            setError(err.message);
        } finally {
            setLoading(false);
            setOptimizationProgress(0);
        }
    };

    const generateRandomParameters = () => {
        const randomParams: Record<string, number | string> = {};
        
        Object.entries(parameterRanges).forEach(([param, [min, max]]) => {
            randomParams[param] = min + Math.random() * (max - min);
            
            // Round to reasonable precision
            if (param.includes('window')) {
                randomParams[param] = Math.round(randomParams[param] as number);
            } else {
                randomParams[param] = Number((randomParams[param] as number).toFixed(3));
            }
        });

        return randomParams;
    };

    const handleParameterChange = (param: string, value: number | string) => {
        setParameters(prev => ({
            ...prev,
            [param]: value
        }));
    };

    const renderParameterControls = () => {
        return Object.entries(parameters).map(([param, value]) => {
            const range = parameterRanges[param];
            if (!range) return null;

            return (
                <Grid item xs={12} md={6} key={param}>
                    <Card>
                        <CardContent>
                            <Typography gutterBottom>
                                {param.replace('_', ' ').toUpperCase()}
                            </Typography>
                            <Slider
                                value={typeof value === 'number' ? value : parseFloat(value)}
                                onChange={(_, newValue) => 
                                    handleParameterChange(param, newValue as number)
                                }
                                min={range[0]}
                                max={range[1]}
                                step={param.includes('window') ? 1 : 0.001}
                                valueLabelDisplay="auto"
                                marks={[
                                    { value: range[0], label: range[0].toFixed(2) },
                                    { value: range[1], label: range[1].toFixed(2) }
                                ]}
                            />
                            <TextField
                                size="small"
                                value={value}
                                onChange={(e) => {
                                    const val = e.target.value;
                                    if (!isNaN(Number(val))) {
                                        handleParameterChange(param, Number(val));
                                    }
                                }}
                                sx={{ mt: 1 }}
                            />
                        </CardContent>
                    </Card>
                </Grid>
            );
        });
    };

    const renderResultsChart = () => {
        if (optimizationResults.length === 0) return null;

        const data = optimizationResults.map((result, index) => ({
            iteration: index + 1,
            ...result.metrics,
            ...result.parameters
        }));

        return (
            <Box sx={{ height: 400, mt: 4 }}>
                <ResponsiveContainer>
                    <LineChart data={data}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis dataKey="iteration" />
                        <YAxis />
                        <Tooltip />
                        <Legend />
                        <Line 
                            type="monotone" 
                            dataKey={selectedMetric} 
                            stroke="#8884d8" 
                            name={selectedMetric.replace('_', ' ')} 
                        />
                        {Object.keys(parameters).map((param, index) => (
                            <Line
                                key={param}
                                type="monotone"
                                dataKey={param}
                                stroke={`hsl(${index * 45}, 70%, 50%)`}
                                name={param.replace('_', ' ')}
                            />
                        ))}
                    </LineChart>
                </ResponsiveContainer>
            </Box>
        );
    };

    const renderResultsTable = () => {
        if (optimizationResults.length === 0) return null;

        return (
            <TableContainer component={Paper} sx={{ mt: 4 }}>
                <Table size="small">
                    <TableHead>
                        <TableRow>
                            <TableCell>Trial</TableCell>
                            {Object.keys(parameters).map(param => (
                                <TableCell key={param}>{param.replace('_', ' ')}</TableCell>
                            ))}
                            {Object.keys(optimizationResults[0].metrics).map(metric => (
                                <TableCell key={metric}>{metric.replace('_', ' ')}</TableCell>
                            ))}
                            <TableCell>Performance Change</TableCell>
                        </TableRow>
                    </TableHead>
                    <TableBody>
                        {optimizationResults.map((result, index) => (
                            <TableRow key={index}>
                                <TableCell>{index + 1}</TableCell>
                                {Object.keys(parameters).map(param => (
                                    <TableCell key={param}>
                                        {result.parameters[param].toFixed(3)}
                                    </TableCell>
                                ))}
                                {Object.entries(result.metrics).map(([metric, value]) => (
                                    <TableCell key={metric}>
                                        {value.toFixed(3)}
                                    </TableCell>
                                ))}
                                <TableCell>
                                    {(result.performance_change * 100).toFixed(1)}%
                                </TableCell>
                            </TableRow>
                        ))}
                    </TableBody>
                </Table>
            </TableContainer>
        );
    };

    return (
        <Paper sx={{ p: 2 }}>
            <Typography variant="h6" gutterBottom>
                Feature Optimization
            </Typography>

            <Grid container spacing={2}>
                <Grid item xs={12}>
                    <Box display="flex" alignItems="center" gap={2}>
                        <FormControlLabel
                            control={
                                <Switch
                                    checked={autoOptimize}
                                    onChange={(e) => setAutoOptimize(e.target.checked)}
                                />
                            }
                            label="Auto-optimize"
                        />
                        <Autocomplete
                            options={['prediction_power', 'correlation', 'mutual_info', 'r2', 'stability']}
                            value={selectedMetric}
                            onChange={(_, newValue) => setSelectedMetric(newValue || 'prediction_power')}
                            renderInput={(params) => (
                                <TextField
                                    {...params}
                                    label="Optimize for"
                                    size="small"
                                    sx={{ width: 200 }}
                                />
                            )}
                            sx={{ width: 200 }}
                        />
                    </Box>
                </Grid>

                {renderParameterControls()}

                <Grid item xs={12}>
                    <Box display="flex" alignItems="center" gap={2}>
                        <Button
                            variant="contained"
                            onClick={optimizeParameters}
                            disabled={loading || Object.keys(parameters).length === 0}
                        >
                            {autoOptimize ? 'Start Auto-Optimization' : 'Test Parameters'}
                        </Button>
                        {loading && (
                            <Box display="flex" alignItems="center" gap={1}>
                                <CircularProgress size={20} />
                                <Typography variant="body2">
                                    {optimizationProgress.toFixed(0)}% Complete
                                </Typography>
                            </Box>
                        )}
                    </Box>
                </Grid>

                {error && (
                    <Grid item xs={12}>
                        <Alert severity="error">{error}</Alert>
                    </Grid>
                )}
            </Grid>

            {renderResultsChart()}
            {renderResultsTable()}
        </Paper>
    );
};

export default FeatureOptimization;