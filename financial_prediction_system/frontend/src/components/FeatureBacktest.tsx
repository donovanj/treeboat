import React, { useState, useEffect } from 'react';
import {
    Paper,
    Typography,
    Grid,
    Box,
    CircularProgress,
    Alert,
    Table,
    TableBody,
    TableCell,
    TableContainer,
    TableHead,
    TableRow,
    Chip,
    Accordion,
    AccordionSummary,
    AccordionDetails,
    LinearProgress
} from '@mui/material';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import { Line } from 'react-chartjs-2';
import { FeatureComposition } from '../types/features';

interface BacktestMetrics {
    feature_metrics: Record<string, number>;
    stability_metrics: Record<string, number>;
    cv_metrics?: Record<string, number>;
    error?: string;
}

interface FeatureBacktestProps {
    composition: FeatureComposition;
    startDate: string;
    endDate: string;
    onComplete?: (results: BacktestMetrics) => void;
}

const FeatureBacktest: React.FC<FeatureBacktestProps> = ({
    composition,
    startDate,
    endDate,
    onComplete
}) => {
    const [results, setResults] = useState<BacktestMetrics | null>(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [timeSeriesData, setTimeSeriesData] = useState<any>(null);

    useEffect(() => {
        const runBacktest = async () => {
            setLoading(true);
            setError(null);

            try {
                const response = await fetch('/api/features/backtest', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        composition_id: composition.id,
                        start_date: startDate,
                        end_date: endDate,
                    }),
                });

                if (!response.ok) {
                    throw new Error('Failed to run backtest');
                }

                const data = await response.json();
                setResults(data);
                setTimeSeriesData(data.time_series);
                
                if (onComplete) {
                    onComplete(data);
                }
            } catch (err: any) {
                setError(err.message);
            } finally {
                setLoading(false);
            }
        };

        runBacktest();
    }, [composition, startDate, endDate]);

    const formatNumber = (num: number) => {
        return Number(num.toFixed(4)).toString();
    };

    const getCorrelationColor = (value: number) => {
        const absValue = Math.abs(value);
        if (absValue > 0.7) return 'success';
        if (absValue > 0.3) return 'warning';
        return 'error';
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

    if (loading) {
        return (
            <Box display="flex" justifyContent="center" alignItems="center" minHeight={400}>
                <CircularProgress />
            </Box>
        );
    }

    if (error) {
        return (
            <Alert severity="error" sx={{ mt: 2 }}>
                {error}
            </Alert>
        );
    }

    if (!results) {
        return null;
    }

    return (
        <Paper sx={{ p: 2 }}>
            <Typography variant="h6" gutterBottom>
                Feature Backtest Results
            </Typography>

            {/* Feature Performance Chart */}
            {timeSeriesData && (
                <Box sx={{ height: 300, mb: 4 }}>
                    <Line
                        data={{
                            labels: timeSeriesData.dates,
                            datasets: [
                                {
                                    label: 'Target',
                                    data: timeSeriesData.target,
                                    borderColor: 'rgb(75, 192, 192)',
                                    yAxisID: 'y',
                                },
                                ...timeSeriesData.features.map((feature: any, index: number) => ({
                                    label: `Feature ${index + 1}`,
                                    data: feature,
                                    borderColor: `hsl(${index * 30}, 70%, 50%)`,
                                    yAxisID: 'y1',
                                })),
                            ],
                        }}
                        options={chartOptions}
                    />
                </Box>
            )}

            {/* Feature Metrics */}
            <Accordion defaultExpanded>
                <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                    <Typography>Feature Metrics</Typography>
                </AccordionSummary>
                <AccordionDetails>
                    <TableContainer>
                        <Table size="small">
                            <TableHead>
                                <TableRow>
                                    <TableCell>Feature</TableCell>
                                    <TableCell align="right">Correlation</TableCell>
                                    <TableCell align="right">Mutual Info</TableCell>
                                    <TableCell align="right">R²</TableCell>
                                </TableRow>
                            </TableHead>
                            <TableBody>
                                {Object.entries(results.feature_metrics)
                                    .filter(([key]) => key.endsWith('correlation'))
                                    .map(([key, value]) => {
                                        const featureId = key.split('_')[1];
                                        return (
                                            <TableRow key={key}>
                                                <TableCell>Feature {featureId}</TableCell>
                                                <TableCell align="right">
                                                    <Chip
                                                        label={formatNumber(value)}
                                                        color={getCorrelationColor(value)}
                                                        size="small"
                                                    />
                                                </TableCell>
                                                <TableCell align="right">
                                                    {formatNumber(results.feature_metrics[`feature_${featureId}_mutual_info`] || 0)}
                                                </TableCell>
                                                <TableCell align="right">
                                                    {formatNumber(results.feature_metrics[`feature_${featureId}_r2`] || 0)}
                                                </TableCell>
                                            </TableRow>
                                        );
                                    })}
                            </TableBody>
                        </Table>
                    </TableContainer>
                </AccordionDetails>
            </Accordion>

            {/* Stability Metrics */}
            <Accordion>
                <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                    <Typography>Stability Analysis</Typography>
                </AccordionSummary>
                <AccordionDetails>
                    <Grid container spacing={2}>
                        {Object.entries(results.stability_metrics)
                            .filter(([key]) => key.endsWith('cv_252d'))
                            .map(([key, value]) => {
                                const featureId = key.split('_')[1];
                                return (
                                    <Grid item xs={12} key={key}>
                                        <Typography variant="body2" gutterBottom>
                                            Feature {featureId} Stability
                                        </Typography>
                                        <Box display="flex" alignItems="center">
                                            <Box width="100%" mr={1}>
                                                <LinearProgress
                                                    variant="determinate"
                                                    value={Math.min(value * 50, 100)}
                                                    color={value > 1.0 ? "warning" : "primary"}
                                                />
                                            </Box>
                                            <Box minWidth={35}>
                                                <Typography variant="body2" color="text.secondary">
                                                    {formatNumber(value)}
                                                </Typography>
                                            </Box>
                                        </Box>
                                    </Grid>
                                );
                            })}
                    </Grid>
                </AccordionDetails>
            </Accordion>

            {/* Cross-Validation Results */}
            {results.cv_metrics && (
                <Accordion>
                    <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                        <Typography>Cross-Validation Results</Typography>
                    </AccordionSummary>
                    <AccordionDetails>
                        <Grid container spacing={2}>
                            {Object.entries(results.cv_metrics).map(([key, value]) => (
                                <Grid item xs={12} md={6} key={key}>
                                    <Typography variant="body2" color="textSecondary">
                                        {key.replace(/_/g, ' ').toUpperCase()}
                                    </Typography>
                                    <Typography variant="h6">
                                        {formatNumber(value)}
                                    </Typography>
                                </Grid>
                            ))}
                        </Grid>
                    </AccordionDetails>
                </Accordion>
            )}
        </Paper>
    );
};

export default FeatureBacktest;