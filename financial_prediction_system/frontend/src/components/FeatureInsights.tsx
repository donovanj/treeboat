import React, { useState, useEffect } from 'react';
import {
    Paper,
    Typography,
    Grid,
    Box,
    Card,
    CardContent,
    List,
    ListItem,
    ListItemIcon,
    ListItemText,
    Divider,
    Chip,
    CircularProgress,
    Alert,
    IconButton,
    Collapse,
    Tooltip
} from '@mui/material';
import {
    TrendingUp,
    Warning,
    CheckCircle,
    Info,
    ExpandMore,
    ExpandLess,
    Lightbulb,
    Timeline,
    Speed,
    Psychology
} from '@mui/icons-material';
import { FeatureComposition } from '../types/features';

interface Insight {
    type: 'success' | 'warning' | 'info' | 'error';
    message: string;
    details?: string;
    recommendations?: string[];
    category: 'statistical' | 'predictive' | 'stability' | 'composition';
}

interface FeatureInsightsProps {
    composition: FeatureComposition;
    metrics: {
        correlation: number;
        mutual_info: number;
        r2: number;
        stability: number;
        prediction_power: number;
    };
    timeSeriesData?: {
        dates: string[];
        values: number[];
    };
}

const FeatureInsights: React.FC<FeatureInsightsProps> = ({
    composition,
    metrics,
    timeSeriesData
}) => {
    const [insights, setInsights] = useState<Insight[]>([]);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [expandedCategories, setExpandedCategories] = useState<Set<string>>(new Set(['statistical']));
    const [selectedInsight, setSelectedInsight] = useState<Insight | null>(null);

    useEffect(() => {
        generateInsights();
    }, [composition, metrics, timeSeriesData]);

    const generateInsights = () => {
        setLoading(true);
        const newInsights: Insight[] = [];

        try {
            // Statistical insights
            if (Math.abs(metrics.correlation) < 0.2) {
                newInsights.push({
                    type: 'warning',
                    category: 'statistical',
                    message: 'Low correlation with target variable',
                    details: 'The feature shows weak linear correlation with the target variable',
                    recommendations: [
                        'Consider non-linear transformations',
                        'Explore feature interactions',
                        'Validate feature calculation logic'
                    ]
                });
            }

            if (metrics.mutual_info > 0.7) {
                newInsights.push({
                    type: 'success',
                    category: 'statistical',
                    message: 'High mutual information score',
                    details: 'Feature captures significant non-linear relationships',
                    recommendations: [
                        'Consider creating ensemble with linear features',
                        'Analyze feature importance in tree-based models'
                    ]
                });
            }

            // Predictive power insights
            if (metrics.r2 > 0.4) {
                newInsights.push({
                    type: 'success',
                    category: 'predictive',
                    message: 'Strong predictive power',
                    details: `R² score of ${metrics.r2.toFixed(3)} indicates good fit`,
                    recommendations: [
                        'Monitor for stability over time',
                        'Consider feature in ensemble models'
                    ]
                });
            }

            // Stability insights
            if (metrics.stability > 1.0) {
                newInsights.push({
                    type: 'warning',
                    category: 'stability',
                    message: 'High feature volatility',
                    details: 'Feature shows significant variation over time',
                    recommendations: [
                        'Apply smoothing techniques',
                        'Consider longer lookback periods',
                        'Implement outlier handling'
                    ]
                });
            }

            // Time series analysis
            if (timeSeriesData && timeSeriesData.values.length > 0) {
                const values = timeSeriesData.values;
                const recentTrend = analyzeTrend(values.slice(-20));
                
                if (Math.abs(recentTrend) > 0.8) {
                    newInsights.push({
                        type: 'info',
                        category: 'stability',
                        message: 'Strong recent trend detected',
                        details: 'Feature shows consistent directional movement',
                        recommendations: [
                            'Consider detrending the feature',
                            'Monitor for regime changes',
                            'Validate trend persistence'
                        ]
                    });
                }
            }

            // Feature composition insights
            const compositionInsights = analyzeComposition(composition);
            newInsights.push(...compositionInsights);

            setInsights(newInsights);

        } catch (err: any) {
            setError(err.message);
        } finally {
            setLoading(false);
        }
    };

    const analyzeTrend = (values: number[]): number => {
        // Simple linear trend analysis
        const n = values.length;
        if (n < 2) return 0;

        const xMean = (n - 1) / 2;
        const yMean = values.reduce((a, b) => a + b, 0) / n;

        let numerator = 0;
        let denominator = 0;

        for (let i = 0; i < n; i++) {
            const x = i - xMean;
            const y = values[i] - yMean;
            numerator += x * y;
            denominator += x * x;
        }

        return denominator ? numerator / denominator : 0;
    };

    const analyzeComposition = (comp: FeatureComposition): Insight[] => {
        const insights: Insight[] = [];

        // Analyze feature complexity
        if (comp.formula && comp.formula.length > 200) {
            insights.push({
                type: 'warning',
                category: 'composition',
                message: 'Complex feature formula',
                details: 'Feature calculation involves complex operations',
                recommendations: [
                    'Break down into simpler components',
                    'Validate each calculation step',
                    'Consider performance implications'
                ]
            });
        }

        // Check for common patterns
        if (comp.formula && comp.formula.includes('rolling')) {
            insights.push({
                type: 'info',
                category: 'composition',
                message: 'Rolling window calculations detected',
                details: 'Feature uses rolling window operations',
                recommendations: [
                    'Validate window size selection',
                    'Consider variable window sizes',
                    'Monitor for lookahead bias'
                ]
            });
        }

        return insights;
    };

    const toggleCategory = (category: string) => {
        setExpandedCategories(prev => {
            const next = new Set(prev);
            if (next.has(category)) {
                next.delete(category);
            } else {
                next.add(category);
            }
            return next;
        });
    };

    const getCategoryIcon = (category: string) => {
        switch (category) {
            case 'statistical':
                return <Timeline />;
            case 'predictive':
                return <Psychology />;
            case 'stability':
                return <Speed />;
            case 'composition':
                return <Lightbulb />;
            default:
                return <Info />;
        }
    };

    const getInsightIcon = (type: string) => {
        switch (type) {
            case 'success':
                return <CheckCircle color="success" />;
            case 'warning':
                return <Warning color="warning" />;
            case 'error':
                return <Warning color="error" />;
            default:
                return <Info color="info" />;
        }
    };

    if (loading) {
        return (
            <Box display="flex" justifyContent="center" alignItems="center" minHeight={200}>
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

    const categories = ['statistical', 'predictive', 'stability', 'composition'];

    return (
        <Paper sx={{ p: 2 }}>
            <Typography variant="h6" gutterBottom>
                Feature Insights
            </Typography>

            {categories.map(category => (
                <Card key={category} sx={{ mb: 2 }}>
                    <CardContent>
                        <Box
                            display="flex"
                            alignItems="center"
                            sx={{ cursor: 'pointer' }}
                            onClick={() => toggleCategory(category)}
                        >
                            <Box display="flex" alignItems="center" flex={1}>
                                {getCategoryIcon(category)}
                                <Typography variant="subtitle1" sx={{ ml: 1, textTransform: 'capitalize' }}>
                                    {category} Analysis
                                </Typography>
                            </Box>
                            <IconButton size="small">
                                {expandedCategories.has(category) ? <ExpandLess /> : <ExpandMore />}
                            </IconButton>
                        </Box>

                        <Collapse in={expandedCategories.has(category)}>
                            <List>
                                {insights
                                    .filter(insight => insight.category === category)
                                    .map((insight, index) => (
                                        <React.Fragment key={index}>
                                            <ListItem
                                                button
                                                onClick={() => setSelectedInsight(
                                                    selectedInsight?.message === insight.message ? null : insight
                                                )}
                                            >
                                                <ListItemIcon>
                                                    {getInsightIcon(insight.type)}
                                                </ListItemIcon>
                                                <ListItemText
                                                    primary={insight.message}
                                                    secondary={selectedInsight?.message === insight.message ? insight.details : null}
                                                />
                                            </ListItem>
                                            
                                            <Collapse in={selectedInsight?.message === insight.message}>
                                                <Box sx={{ pl: 9, pr: 2, pb: 2 }}>
                                                    {insight.recommendations && (
                                                        <>
                                                            <Typography variant="subtitle2" gutterBottom>
                                                                Recommendations:
                                                            </Typography>
                                                            <List dense>
                                                                {insight.recommendations.map((rec, idx) => (
                                                                    <ListItem key={idx}>
                                                                        <ListItemIcon sx={{ minWidth: 30 }}>
                                                                            <Info fontSize="small" color="primary" />
                                                                        </ListItemIcon>
                                                                        <ListItemText primary={rec} />
                                                                    </ListItem>
                                                                ))}
                                                            </List>
                                                        </>
                                                    )}
                                                </Box>
                                            </Collapse>
                                            
                                            {index < insights.filter(i => i.category === category).length - 1 && (
                                                <Divider variant="inset" component="li" />
                                            )}
                                        </React.Fragment>
                                    ))}
                            </List>
                        </Collapse>
                    </CardContent>
                </Card>
            ))}
        </Paper>
    );
};

export default FeatureInsights;