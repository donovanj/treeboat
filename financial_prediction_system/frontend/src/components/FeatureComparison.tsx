import React, { useState, useEffect } from 'react';
import {
    Paper,
    Typography,
    Grid,
    Box,
    Table,
    TableBody,
    TableCell,
    TableContainer,
    TableHead,
    TableRow,
    Chip,
    CircularProgress,
    Alert,
    Card,
    CardContent,
    Button,
    Select,
    MenuItem,
    FormControl,
    InputLabel
} from '@mui/material';
import { Scatter } from 'react-chartjs-2';
import { FeatureComposition } from '../types/features';

interface ComparisonMetrics {
    correlation: number;
    mutual_info: number;
    r2: number;
    stability: number;
    prediction_power: number;
}

interface FeatureComparisonProps {
    compositions: FeatureComposition[];
    startDate: string;
    endDate: string;
    onInsightFound?: (insight: string) => void;
}

const FeatureComparison: React.FC<FeatureComparisonProps> = ({
    compositions,
    startDate,
    endDate,
    onInsightFound
}) => {
    const [metrics, setMetrics] = useState<Record<string, ComparisonMetrics>>({});
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [selectedMetric, setSelectedMetric] = useState<string>('correlation');
    const [correlationData, setCorrelationData] = useState<any>(null);

    useEffect(() => {
        const fetchComparisonData = async () => {
            setLoading(true);
            setError(null);

            try {
                const promises = compositions.map(async (comp) => {
                    const response = await fetch('/api/features/backtest/performance', {
                        method: 'GET',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        params: {
                            symbol: comp.symbol,
                            start_date: startDate,
                            end_date: endDate
                        }
                    });

                    if (!response.ok) {
                        throw new Error(`Failed to fetch metrics for ${comp.name}`);
                    }

                    const data = await response.json();
                    return {
                        id: comp.id,
                        metrics: data[comp.id].metrics
                    };
                });

                const results = await Promise.all(promises);
                const metricsMap: Record<string, ComparisonMetrics> = {};
                
                results.forEach(result => {
                    metricsMap[result.id] = result.metrics;
                });

                setMetrics(metricsMap);
                
                // Generate correlation matrix
                const correlations = generateCorrelationMatrix(results);
                setCorrelationData(correlations);
                
                // Find insights
                const insights = analyzeResults(metricsMap);
                if (onInsightFound && insights.length > 0) {
                    insights.forEach(insight => onInsightFound(insight));
                }

            } catch (err: any) {
                setError(err.message);
            } finally {
                setLoading(false);
            }
        };

        if (compositions.length > 0) {
            fetchComparisonData();
        }
    }, [compositions, startDate, endDate]);

    const generateCorrelationMatrix = (results: any[]) => {
        const features = results.map(r => ({
            x: r.metrics.correlation,
            y: r.metrics.mutual_info,
            r: r.metrics.r2 * 20, // Scale R² for bubble size
            label: compositions.find(c => c.id === r.id)?.name || r.id
        }));

        return {
            datasets: [{
                label: 'Feature Correlations',
                data: features,
                backgroundColor: features.map((_, i) => 
                    `hsla(${i * 360 / features.length}, 70%, 50%, 0.6)`
                )
            }]
        };
    };

    const analyzeResults = (
        metricsMap: Record<string, ComparisonMetrics>
    ): string[] => {
        const insights: string[] = [];
        
        // Find best performing features
        const bestCorrelation = Object.entries(metricsMap)
            .reduce((best, [id, metrics]) => 
                Math.abs(metrics.correlation) > Math.abs(best.correlation)
                    ? { id, correlation: metrics.correlation }
                    : best,
                { id: '', correlation: 0 }
            );

        const composition = compositions.find(c => c.id === bestCorrelation.id);
        if (composition) {
            insights.push(
                `Feature composition "${composition.name}" shows strongest correlation with target`
            );
        }

        // Identify complementary features
        const complementaryPairs = findComplementaryFeatures(metricsMap);
        complementaryPairs.forEach(pair => {
            const comp1 = compositions.find(c => c.id === pair[0]);
            const comp2 = compositions.find(c => c.id === pair[1]);
            if (comp1 && comp2) {
                insights.push(
                    `Features "${comp1.name}" and "${comp2.name}" show complementary predictive power`
                );
            }
        });

        return insights;
    };

    const findComplementaryFeatures = (
        metricsMap: Record<string, ComparisonMetrics>
    ): [string, string][] => {
        const pairs: [string, string][] = [];
        const ids = Object.keys(metricsMap);

        for (let i = 0; i < ids.length; i++) {
            for (let j = i + 1; j < ids.length; j++) {
                const metrics1 = metricsMap[ids[i]];
                const metrics2 = metricsMap[ids[j]];

                // Check if features capture different aspects
                if (
                    Math.abs(metrics1.correlation - metrics2.correlation) > 0.3 &&
                    metrics1.mutual_info + metrics2.mutual_info > 1.0
                ) {
                    pairs.push([ids[i], ids[j]]);
                }
            }
        }

        return pairs;
    };

    const formatMetricValue = (value: number) => {
        return Number(value.toFixed(4)).toString();
    };

    const getMetricColor = (value: number, metric: string) => {
        const absValue = Math.abs(value);
        
        if (metric === 'correlation' || metric === 'mutual_info') {
            if (absValue > 0.7) return 'success';
            if (absValue > 0.3) return 'warning';
            return 'error';
        }
        
        if (metric === 'r2') {
            if (absValue > 0.5) return 'success';
            if (absValue > 0.2) return 'warning';
            return 'error';
        }
        
        if (metric === 'stability') {
            if (absValue < 0.5) return 'success';
            if (absValue < 1.0) return 'warning';
            return 'error';
        }
        
        return 'default';
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

    return (
        <Paper sx={{ p: 2 }}>
            <Typography variant="h6" gutterBottom>
                Feature Composition Comparison
            </Typography>

            {/* Metric Selector */}
            <Box sx={{ mb: 3 }}>
                <FormControl sx={{ minWidth: 200 }}>
                    <InputLabel>Compare By</InputLabel>
                    <Select
                        value={selectedMetric}
                        onChange={(e) => setSelectedMetric(e.target.value)}
                        label="Compare By"
                    >
                        <MenuItem value="correlation">Correlation</MenuItem>
                        <MenuItem value="mutual_info">Mutual Information</MenuItem>
                        <MenuItem value="r2">R² Score</MenuItem>
                        <MenuItem value="stability">Stability</MenuItem>
                        <MenuItem value="prediction_power">Prediction Power</MenuItem>
                    </Select>
                </FormControl>
            </Box>

            {/* Correlation Scatter Plot */}
            {correlationData && (
                <Box sx={{ height: 400, mb: 4 }}>
                    <Scatter
                        data={correlationData}
                        options={{
                            responsive: true,
                            plugins: {
                                legend: {
                                    display: false
                                },
                                tooltip: {
                                    callbacks: {
                                        label: (context: any) => {
                                            const point = correlationData.datasets[0].data[context.dataIndex];
                                            return [
                                                `${point.label}`,
                                                `Correlation: ${formatMetricValue(point.x)}`,
                                                `Mutual Info: ${formatMetricValue(point.y)}`,
                                                `R²: ${formatMetricValue(point.r / 20)}`
                                            ];
                                        }
                                    }
                                }
                            },
                            scales: {
                                x: {
                                    title: {
                                        display: true,
                                        text: 'Correlation with Target'
                                    }
                                },
                                y: {
                                    title: {
                                        display: true,
                                        text: 'Mutual Information'
                                    }
                                }
                            }
                        }}
                    />
                </Box>
            )}

            {/* Metrics Table */}
            <TableContainer>
                <Table size="small">
                    <TableHead>
                        <TableRow>
                            <TableCell>Feature Composition</TableCell>
                            <TableCell align="right">Correlation</TableCell>
                            <TableCell align="right">Mutual Info</TableCell>
                            <TableCell align="right">R²</TableCell>
                            <TableCell align="right">Stability</TableCell>
                            <TableCell align="right">Prediction Power</TableCell>
                        </TableRow>
                    </TableHead>
                    <TableBody>
                        {compositions.map(comp => {
                            const compMetrics = metrics[comp.id];
                            if (!compMetrics) return null;

                            return (
                                <TableRow key={comp.id}>
                                    <TableCell>{comp.name}</TableCell>
                                    <TableCell align="right">
                                        <Chip
                                            label={formatMetricValue(compMetrics.correlation)}
                                            color={getMetricColor(compMetrics.correlation, 'correlation')}
                                            size="small"
                                        />
                                    </TableCell>
                                    <TableCell align="right">
                                        <Chip
                                            label={formatMetricValue(compMetrics.mutual_info)}
                                            color={getMetricColor(compMetrics.mutual_info, 'mutual_info')}
                                            size="small"
                                        />
                                    </TableCell>
                                    <TableCell align="right">
                                        <Chip
                                            label={formatMetricValue(compMetrics.r2)}
                                            color={getMetricColor(compMetrics.r2, 'r2')}
                                            size="small"
                                        />
                                    </TableCell>
                                    <TableCell align="right">
                                        <Chip
                                            label={formatMetricValue(compMetrics.stability)}
                                            color={getMetricColor(compMetrics.stability, 'stability')}
                                            size="small"
                                        />
                                    </TableCell>
                                    <TableCell align="right">
                                        <Chip
                                            label={formatMetricValue(compMetrics.prediction_power)}
                                            color={getMetricColor(compMetrics.prediction_power, 'prediction_power')}
                                            size="small"
                                        />
                                    </TableCell>
                                </TableRow>
                            );
                        })}
                    </TableBody>
                </Table>
            </TableContainer>
        </Paper>
    );
};

export default FeatureComparison;