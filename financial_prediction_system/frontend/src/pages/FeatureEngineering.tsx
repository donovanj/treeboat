import React, { useState, useEffect } from 'react';
import { 
    Typography, 
    Paper, 
    Grid, 
    Alert,
    CircularProgress,
    Box,
    Tab,
    Tabs,
    Divider
} from '@mui/material';
import { useGlobalSelection } from '../context/GlobalSelectionContext';
import StockAutocomplete from '../components/StockAutocomplete';
import FeaturePreviewPlot from '../components/FeaturePreviewPlot';
import FeatureMetrics from '../components/FeatureMetrics';
import FeatureBuilder from '../components/FeatureBuilder';
import SavedFeatures from '../components/SavedFeatures';
import TargetPreview from '../components/TargetPreview';
import { useFeatureEngineering } from '../hooks/useFeatureEngineering';

interface TabPanelProps {
    children?: React.ReactNode;
    index: number;
    value: number;
}

function TabPanel(props: TabPanelProps) {
    const { children, value, index, ...other } = props;
    return (
        <div
            role="tabpanel"
            hidden={value !== index}
            id={`feature-tabpanel-${index}`}
            aria-labelledby={`feature-tab-${index}`}
            {...other}
        >
            {value === index && (
                <Box sx={{ p: 3 }}>
                    {children}
                </Box>
            )}
        </div>
    );
}

const FeatureEngineering: React.FC = () => {
    const { stock, startDate, endDate } = useGlobalSelection();
    const [tabValue, setTabValue] = useState(0);
    const [baseData, setBaseData] = useState<any>(null);
    const [currentFeature, setCurrentFeature] = useState<any>(null);
    const [currentTarget, setCurrentTarget] = useState<any>(null);
    const [predefinedFeatures, setPredefinedFeatures] = useState<string[]>([]);
    const [selectedFeature, setSelectedFeature] = useState<any>(null);
    
    const { 
        previewFeature, 
        saveFeature, 
        previewTarget,
        previewData, 
        isLoading, 
        error 
    } = useFeatureEngineering();

    useEffect(() => {
        if (stock && startDate && endDate) {
            fetchData();
        }
    }, [stock, startDate, endDate]);

    const fetchData = async () => {
        try {
            const response = await fetch(`/api/data?symbol=${stock}&start=${startDate}&end=${endDate}`);
            const data = await response.json();
            setBaseData(data);
            setPredefinedFeatures(['open', 'high', 'low', 'close', 'volume']);
        } catch (err: any) {
            console.error('Error fetching data:', err);
        }
    };

    const handlePreview = async (formula: string) => {
        if (!baseData) return;
        
        const result = await previewFeature(formula, baseData);
        if (result) {
            setCurrentFeature({
                name: 'Preview Feature',
                values: result.feature,
                metrics: result.metrics
            });
        }
    };

    const handleTargetChange = async (targetType: string, params: any) => {
        if (!currentFeature?.values || !baseData) return;

        const result = await previewTarget(currentFeature.values, baseData, targetType, params);
        if (result) {
            setCurrentTarget({
                name: targetType,
                values: result.target,
                metrics: result.metrics
            });
        }
    };

    const handleSave = async (name: string, formula: string, type: string) => {
        if (!stock) return;
        
        const result = await saveFeature(name, formula, type, stock);
        if (result) {
            setPredefinedFeatures(prev => [...prev, name]);
        }
    };

    const handleTabChange = (event: React.SyntheticEvent, newValue: number) => {
        setTabValue(newValue);
    };

    const handleEditFeature = (feature: any) => {
        setSelectedFeature(feature);
        setTabValue(0); // Switch to Feature Builder tab
        handlePreview(feature.formula);
    };

    return (
        <Paper sx={{ p: 3, minHeight: '80vh' }}>
            <Typography variant="h4" gutterBottom>
                Feature Engineering Workbench
            </Typography>

            <Grid container spacing={3}>
                {/* Stock Selection */}
                <Grid item xs={12} md={6}>
                    <StockAutocomplete />
                </Grid>

                {/* Status Messages */}
                {isLoading && (
                    <Grid item xs={12}>
                        <Box display="flex" justifyContent="center">
                            <CircularProgress />
                        </Box>
                    </Grid>
                )}
                {error && (
                    <Grid item xs={12}>
                        <Alert severity="error">{error}</Alert>
                    </Grid>
                )}

                {/* Main Content */}
                {!isLoading && !error && (
                    <>
                        {/* Tabs */}
                        <Grid item xs={12}>
                            <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
                                <Tabs value={tabValue} onChange={handleTabChange}>
                                    <Tab label="Feature Builder" />
                                    <Tab label="Analysis & Preview" />
                                    <Tab label="Saved Features" />
                                </Tabs>
                            </Box>
                        </Grid>

                        {/* Tab Panels */}
                        <Grid item xs={12}>
                            <TabPanel value={tabValue} index={0}>
                                <Grid container spacing={3}>
                                    <Grid item xs={12} md={6}>
                                        <FeatureBuilder
                                            onPreview={handlePreview}
                                            onSave={handleSave}
                                            predefinedFeatures={predefinedFeatures}
                                            initialFeature={selectedFeature}
                                        />
                                    </Grid>
                                    <Grid item xs={12} md={6}>
                                        <Box sx={{ mb: 3 }}>
                                            <FeaturePreviewPlot
                                                baseData={baseData}
                                                featureDefinition={currentFeature?.name}
                                                targetDefinition={currentTarget?.name}
                                                featureData={currentFeature?.values}
                                                targetData={currentTarget?.values}
                                            />
                                        </Box>
                                        {currentFeature && (
                                            <TargetPreview
                                                baseData={baseData}
                                                featureValues={currentFeature.values}
                                                onTargetChange={handleTargetChange}
                                            />
                                        )}
                                    </Grid>
                                </Grid>
                            </TabPanel>

                            <TabPanel value={tabValue} index={1}>
                                <Grid container spacing={3}>
                                    <Grid item xs={12}>
                                        {currentFeature && (
                                            <FeatureMetrics
                                                feature={{
                                                    name: currentFeature.name,
                                                    values: currentFeature.values
                                                }}
                                                target={currentTarget && {
                                                    name: currentTarget.name,
                                                    values: currentTarget.values
                                                }}
                                                correlations={currentTarget?.metrics ? {
                                                    pearson: currentTarget.metrics.correlation || 0,
                                                    spearman: currentTarget.metrics.rank_correlation || 0,
                                                    kendall: currentTarget.metrics.mutual_info || 0
                                                } : undefined}
                                                predictivePower={currentTarget?.metrics?.correlation 
                                                    ? Math.abs(currentTarget.metrics.correlation) 
                                                    : undefined}
                                            />
                                        )}
                                    </Grid>
                                </Grid>
                            </TabPanel>

                            <TabPanel value={tabValue} index={2}>
                                {stock ? (
                                    <SavedFeatures
                                        symbol={stock}
                                        onPreview={handlePreview}
                                        onEdit={handleEditFeature}
                                    />
                                ) : (
                                    <Typography>
                                        Select a stock to view saved features
                                    </Typography>
                                )}
                            </TabPanel>
                        </Grid>
                    </>
                )}
            </Grid>
        </Paper>
    );
};

export default FeatureEngineering;
