import React, { useState, useEffect } from 'react';
import { 
    Typography, 
    Paper, 
    Grid, 
    Box, 
    Tab, 
    Tabs,
    Button,
    Alert,
    CircularProgress,
    Card,
    CardContent,
    IconButton,
    List,
    ListItem,
    ListItemText,
    ListItemSecondaryAction,
    Divider
} from '@mui/material';
import { useGlobalSelection } from '../context/GlobalSelectionContext';
import { useStockFeature } from '../context/StockFeatureContext';
import SavedFeatures from '../components/SavedFeatures';
import TargetEngineering from '../components/TargetEngineering';
import ChevronRightIcon from '@mui/icons-material/ChevronRight';
import DeleteIcon from '@mui/icons-material/Delete';

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
            id={`target-tabpanel-${index}`}
            aria-labelledby={`target-tab-${index}`}
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

const TargetConstruction: React.FC = () => {
    const { stock, startDate, endDate } = useGlobalSelection();
    const { selectedFeatures } = useStockFeature();
    const [tabValue, setTabValue] = useState(0);
    const [baseData, setBaseData] = useState<any>(null);
    const [selectedFeatureIds, setSelectedFeatureIds] = useState<string[]>([]);
    const [targets, setTargets] = useState<any[]>([]);
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);

    useEffect(() => {
        if (stock && startDate && endDate) {
            fetchData();
        }
    }, [stock, startDate, endDate]);

    const fetchData = async () => {
        setIsLoading(true);
        setError(null);
        
        try {
            const response = await fetch(`/api/data?symbol=${stock}&start=${startDate}&end=${endDate}`);
            const data = await response.json();
            setBaseData(data);
            
            // Also fetch saved features for this stock
            const featuresResponse = await fetch(`/api/features?symbol=${stock}`);
            if (featuresResponse.ok) {
                // Process features if needed
            }
        } catch (err: any) {
            setError(err.message || 'Failed to fetch data');
        } finally {
            setIsLoading(false);
        }
    };

    const handleTabChange = (event: React.SyntheticEvent, newValue: number) => {
        setTabValue(newValue);
    };

    const handleFeatureSelect = (featureId: string) => {
        setSelectedFeatureIds(prev => 
            prev.includes(featureId) ? 
                prev.filter(id => id !== featureId) : 
                [...prev, featureId]
        );
    };

    const handleTargetSave = (target: any) => {
        setTargets(prev => [...prev, {
            ...target,
            id: `target_${Date.now()}`
        }]);
    };

    const handleTargetDelete = (targetId: string) => {
        setTargets(prev => prev.filter(t => t.id !== targetId));
    };

    const handleExportToModelTraining = async () => {
        if (targets.length === 0) {
            setError('No targets to export');
            return;
        }

        setIsLoading(true);
        setError(null);
        
        try {
            const response = await fetch('/api/targets/export', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    symbol: stock,
                    targets: targets.map(t => ({
                        name: t.name,
                        type: t.type,
                        parameters: t.parameters
                    })),
                    selected_features: selectedFeatureIds
                }),
            });

            if (!response.ok) {
                throw new Error('Failed to export targets');
            }

            // Successfully exported
            // You could navigate to the model training page here if needed
        } catch (err: any) {
            setError(err.message || 'Failed to export targets');
        } finally {
            setIsLoading(false);
        }
    };

    return (
        <Paper sx={{ p: 3, minHeight: '80vh' }}>
            <Typography variant="h4" gutterBottom>
                Target Construction
            </Typography>

            {isLoading && (
                <Box display="flex" justifyContent="center" p={3}>
                    <CircularProgress />
                </Box>
            )}
            
            {error && (
                <Alert severity="error" sx={{ mb: 3 }}>
                    {error}
                </Alert>
            )}

            <Tabs value={tabValue} onChange={handleTabChange} sx={{ mb: 3 }}>
                <Tab label="Select Features" />
                <Tab label="Create Targets" />
                <Tab label="Export to Model" />
            </Tabs>

            <TabPanel value={tabValue} index={0}>
                <Typography variant="h6" gutterBottom>
                    Select Features for Target Construction
                </Typography>
                
                <Grid container spacing={3}>
                    <Grid item xs={12} md={6}>
                        <Typography variant="subtitle1" gutterBottom>
                            Available Features
                        </Typography>
                        
                        {stock ? (
                            <SavedFeatures
                                symbol={stock}
                                onSelect={handleFeatureSelect}
                                selectedIds={selectedFeatureIds}
                                selectionMode={true}
                            />
                        ) : (
                            <Alert severity="info">
                                Please select a stock to view features
                            </Alert>
                        )}
                    </Grid>
                    
                    <Grid item xs={12} md={6}>
                        <Typography variant="subtitle1" gutterBottom>
                            Selected Features
                        </Typography>
                        
                        {selectedFeatureIds.length > 0 ? (
                            <Card variant="outlined">
                                <List>
                                    {selectedFeatureIds.map((id) => (
                                        <ListItem key={id}>
                                            <ListItemText 
                                                primary={id}
                                                secondary="Feature" 
                                            />
                                            <ListItemSecondaryAction>
                                                <IconButton 
                                                    edge="end" 
                                                    onClick={() => handleFeatureSelect(id)}
                                                >
                                                    <DeleteIcon />
                                                </IconButton>
                                            </ListItemSecondaryAction>
                                        </ListItem>
                                    ))}
                                </List>
                            </Card>
                        ) : (
                            <Alert severity="info">
                                No features selected. Select features from the list.
                            </Alert>
                        )}
                        
                        <Box sx={{ mt: 3, display: 'flex', justifyContent: 'flex-end' }}>
                            <Button 
                                variant="contained" 
                                endIcon={<ChevronRightIcon />}
                                onClick={() => setTabValue(1)}
                                disabled={selectedFeatureIds.length === 0}
                            >
                                Continue to Target Creation
                            </Button>
                        </Box>
                    </Grid>
                </Grid>
            </TabPanel>

            <TabPanel value={tabValue} index={1}>
                <Typography variant="h6" gutterBottom>
                    Create Target Variables
                </Typography>
                
                {baseData ? (
                    <TargetEngineering
                        baseData={baseData}
                        onTargetChange={(target) => console.log('Target changed:', target)}
                    />
                ) : (
                    <Alert severity="info">
                        Loading data...
                    </Alert>
                )}
                
                <Divider sx={{ my: 3 }} />
                
                <Typography variant="h6" gutterBottom>
                    Saved Targets
                </Typography>
                
                {targets.length > 0 ? (
                    <Grid container spacing={2}>
                        {targets.map((target) => (
                            <Grid item xs={12} md={6} key={target.id}>
                                <Card variant="outlined">
                                    <CardContent>
                                        <Box display="flex" justifyContent="space-between" alignItems="center">
                                            <Typography variant="subtitle1">
                                                {target.name || 'Unnamed Target'}
                                            </Typography>
                                            <IconButton 
                                                size="small" 
                                                onClick={() => handleTargetDelete(target.id)}
                                            >
                                                <DeleteIcon />
                                            </IconButton>
                                        </Box>
                                        <Typography variant="body2" color="text.secondary">
                                            Type: {target.type}
                                        </Typography>
                                        <Typography variant="body2" color="text.secondary">
                                            Horizon: {target.parameters?.horizon || 'N/A'}
                                        </Typography>
                                    </CardContent>
                                </Card>
                            </Grid>
                        ))}
                    </Grid>
                ) : (
                    <Alert severity="info">
                        No targets saved yet
                    </Alert>
                )}
                
                <Box sx={{ mt: 3, display: 'flex', justifyContent: 'flex-end' }}>
                    <Button 
                        variant="contained" 
                        endIcon={<ChevronRightIcon />}
                        onClick={() => setTabValue(2)}
                        disabled={targets.length === 0}
                    >
                        Continue to Export
                    </Button>
                </Box>
            </TabPanel>

            <TabPanel value={tabValue} index={2}>
                <Typography variant="h6" gutterBottom>
                    Export to Model Training
                </Typography>
                
                <Grid container spacing={3}>
                    <Grid item xs={12} md={6}>
                        <Typography variant="subtitle1" gutterBottom>
                            Selected Features ({selectedFeatureIds.length})
                        </Typography>
                        <Card variant="outlined" sx={{ mb: 3 }}>
                            <List dense>
                                {selectedFeatureIds.map((id) => (
                                    <ListItem key={id}>
                                        <ListItemText primary={id} />
                                    </ListItem>
                                ))}
                            </List>
                        </Card>
                    </Grid>
                    
                    <Grid item xs={12} md={6}>
                        <Typography variant="subtitle1" gutterBottom>
                            Targets ({targets.length})
                        </Typography>
                        <Card variant="outlined" sx={{ mb: 3 }}>
                            <List dense>
                                {targets.map((target) => (
                                    <ListItem key={target.id}>
                                        <ListItemText 
                                            primary={target.name || 'Unnamed Target'} 
                                            secondary={`Type: ${target.type}`}
                                        />
                                    </ListItem>
                                ))}
                            </List>
                        </Card>
                    </Grid>
                    
                    <Grid item xs={12}>
                        <Box sx={{ display: 'flex', justifyContent: 'center' }}>
                            <Button 
                                variant="contained" 
                                color="primary" 
                                size="large"
                                onClick={handleExportToModelTraining}
                                disabled={targets.length === 0 || selectedFeatureIds.length === 0}
                            >
                                Export to Model Training
                            </Button>
                        </Box>
                    </Grid>
                </Grid>
            </TabPanel>
        </Paper>
    );
};

export default TargetConstruction;