import React from 'react';
import { Paper, Typography, Table, TableBody, TableCell, TableContainer, TableHead, TableRow } from '@mui/material';
import Plot from 'react-plotly.js';

interface FeatureMetricsProps {
    feature: {
        name: string;
        values: number[];
    };
    target?: {
        name: string;
        values: number[];
    };
    correlations?: {
        pearson: number;
        spearman: number;
        kendall: number;
    };
    predictivePower?: number;
}

const FeatureMetrics: React.FC<FeatureMetricsProps> = ({
    feature,
    target,
    correlations,
    predictivePower
}) => {
    if (!feature?.values?.length) {
        return (
            <Paper sx={{ p: 2, textAlign: 'center' }}>
                <Typography color="text.secondary">
                    Generate a feature to see metrics
                </Typography>
            </Paper>
        );
    }

    const calculateStats = (values: number[]) => {
        const sorted = [...values].sort((a, b) => a - b);
        return {
            mean: values.reduce((a, b) => a + b, 0) / values.length,
            median: sorted[Math.floor(values.length / 2)],
            std: Math.sqrt(values.reduce((a, b) => a + Math.pow(b - values.reduce((c, d) => c + d, 0) / values.length, 2), 0) / values.length),
            min: sorted[0],
            max: sorted[sorted.length - 1]
        };
    };

    const featureStats = calculateStats(feature.values);

    return (
        <Paper sx={{ p: 2 }}>
            <Typography variant="h6" gutterBottom>
                Feature Analysis
            </Typography>
            
            {/* Statistical Summary */}
            <TableContainer component={Paper} sx={{ mb: 2 }}>
                <Table size="small">
                    <TableHead>
                        <TableRow>
                            <TableCell>Metric</TableCell>
                            <TableCell align="right">Value</TableCell>
                        </TableRow>
                    </TableHead>
                    <TableBody>
                        <TableRow>
                            <TableCell>Mean</TableCell>
                            <TableCell align="right">{featureStats.mean.toFixed(4)}</TableCell>
                        </TableRow>
                        <TableRow>
                            <TableCell>Median</TableCell>
                            <TableCell align="right">{featureStats.median.toFixed(4)}</TableCell>
                        </TableRow>
                        <TableRow>
                            <TableCell>Std Dev</TableCell>
                            <TableCell align="right">{featureStats.std.toFixed(4)}</TableCell>
                        </TableRow>
                        <TableRow>
                            <TableCell>Min</TableCell>
                            <TableCell align="right">{featureStats.min.toFixed(4)}</TableCell>
                        </TableRow>
                        <TableRow>
                            <TableCell>Max</TableCell>
                            <TableCell align="right">{featureStats.max.toFixed(4)}</TableCell>
                        </TableRow>
                    </TableBody>
                </Table>
            </TableContainer>

            {/* Correlation Metrics */}
            {correlations && (
                <>
                    <Typography variant="subtitle1" gutterBottom>
                        Target Correlations
                    </Typography>
                    <TableContainer component={Paper} sx={{ mb: 2 }}>
                        <Table size="small">
                            <TableHead>
                                <TableRow>
                                    <TableCell>Type</TableCell>
                                    <TableCell align="right">Value</TableCell>
                                </TableRow>
                            </TableHead>
                            <TableBody>
                                <TableRow>
                                    <TableCell>Pearson</TableCell>
                                    <TableCell align="right">{correlations.pearson.toFixed(4)}</TableCell>
                                </TableRow>
                                <TableRow>
                                    <TableCell>Spearman</TableCell>
                                    <TableCell align="right">{correlations.spearman.toFixed(4)}</TableCell>
                                </TableRow>
                                <TableRow>
                                    <TableCell>Kendall</TableCell>
                                    <TableCell align="right">{correlations.kendall.toFixed(4)}</TableCell>
                                </TableRow>
                            </TableBody>
                        </Table>
                    </TableContainer>
                </>
            )}

            {/* Distribution Plot */}
            <Plot
                data={[
                    {
                        type: 'histogram',
                        x: feature.values,
                        name: 'Distribution',
                        xbins: {
                            start: Math.min(...feature.values),
                            end: Math.max(...feature.values),
                            size: (Math.max(...feature.values) - Math.min(...feature.values)) / 30
                        }
                    }
                ]}
                layout={{
                    title: 'Feature Distribution',
                    xaxis: { title: 'Value' },
                    yaxis: { title: 'Frequency' },
                    height: 300
                }}
                style={{ width: '100%' }}
            />

            {/* Target Relationship Plot */}
            {target && (
                <Plot
                    data={[
                        {
                            type: 'scatter',
                            mode: 'markers',
                            x: feature.values,
                            y: target.values,
                            name: 'Feature vs Target',
                            marker: { 
                                size: 6,
                                opacity: 0.6
                            }
                        }
                    ]}
                    layout={{
                        title: 'Feature vs Target',
                        xaxis: { title: feature.name },
                        yaxis: { title: target.name },
                        height: 300
                    }}
                    style={{ width: '100%' }}
                />
            )}

            {/* Predictive Power */}
            {predictivePower !== undefined && (
                <Typography variant="body1" sx={{ mt: 2 }}>
                    Estimated Predictive Power: {(predictivePower * 100).toFixed(2)}%
                </Typography>
            )}
        </Paper>
    );
};

export default FeatureMetrics;