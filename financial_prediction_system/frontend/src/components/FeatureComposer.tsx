import React, { useState } from 'react';
import {
    Box,
    Card,
    CardContent,
    Typography,
    TextField,
    Button,
    Stack,
    IconButton,
    Chip,
    Dialog,
    DialogTitle,
    DialogContent,
    DialogActions,
    Alert
} from '@mui/material';
import DeleteIcon from '@mui/icons-material/Delete';
import AddIcon from '@mui/icons-material/Add';

interface Parameter {
    name: string;
    type: string;
    description: string;
    defaultValue: any;
    range?: [number, number];
}

interface Feature {
    name: string;
    description: string;
    parameters: Parameter[];
}

interface ComposedFeature extends Omit<Feature, 'parameters'> {
    id: string;
    parameters: Record<string, any>;
}

interface FeatureComposerProps {
    selectedFeatures: ComposedFeature[];
    onAddFeature: (feature: Feature) => void;
    onRemoveFeature: (id: string) => void;
    onUpdateParameters: (id: string, params: Record<string, any>) => void;
    onPreview: () => void;
    onSave: () => void;
}

const FeatureComposer: React.FC<FeatureComposerProps> = ({
    selectedFeatures,
    onAddFeature,
    onRemoveFeature,
    onUpdateParameters,
    onPreview,
    onSave
}) => {
    const [editingFeature, setEditingFeature] = useState<ComposedFeature | null>(null);

    const handleParameterChange = (paramName: string, value: any) => {
        if (!editingFeature) return;

        const newParams = {
            ...editingFeature.parameters,
            [paramName]: value
        };

        onUpdateParameters(editingFeature.id, newParams);
    };

    const handleCloseDialog = () => {
        setEditingFeature(null);
    };

    return (
        <Box>
            <Typography variant="h6" gutterBottom>
                Composed Features
            </Typography>

            {selectedFeatures.length === 0 ? (
                <Alert severity="info">
                    Select features from the categories to begin composition
                </Alert>
            ) : (
                <Stack spacing={2}>
                    {selectedFeatures.map((feature) => (
                        <Card key={feature.id} variant="outlined">
                            <CardContent>
                                <Box display="flex" justifyContent="space-between" alignItems="center">
                                    <Typography variant="subtitle1">
                                        {feature.name}
                                    </Typography>
                                    <Box>
                                        <IconButton
                                            size="small"
                                            onClick={() => setEditingFeature(feature)}
                                        >
                                            <AddIcon />
                                        </IconButton>
                                        <IconButton
                                            size="small"
                                            onClick={() => onRemoveFeature(feature.id)}
                                        >
                                            <DeleteIcon />
                                        </IconButton>
                                    </Box>
                                </Box>
                                
                                {feature.parameters && Object.entries(feature.parameters).length > 0 && (
                                    <Box sx={{ mt: 1 }}>
                                        {Object.entries(feature.parameters).map(([param, value]) => (
                                            <Chip
                                                key={param}
                                                label={`${param}: ${value}`}
                                                size="small"
                                                sx={{ mr: 1, mb: 1 }}
                                            />
                                        ))}
                                    </Box>
                                )}
                            </CardContent>
                        </Card>
                    ))}

                    <Box display="flex" gap={2} justifyContent="flex-end">
                        <Button 
                            variant="outlined" 
                            onClick={onPreview}
                            disabled={selectedFeatures.length === 0}
                        >
                            Preview
                        </Button>
                        <Button 
                            variant="contained" 
                            onClick={onSave}
                            disabled={selectedFeatures.length === 0}
                        >
                            Save Composition
                        </Button>
                    </Box>
                </Stack>
            )}

            {/* Parameter Configuration Dialog */}
            <Dialog open={!!editingFeature} onClose={handleCloseDialog}>
                <DialogTitle>Configure {editingFeature?.name}</DialogTitle>
                <DialogContent>
                    {editingFeature && Object.entries(editingFeature.parameters).map(([paramName, paramValue]) => (
                        <TextField
                            key={paramName}
                            label={paramName}
                            type={typeof paramValue === 'number' ? 'number' : 'text'}
                            value={paramValue}
                            onChange={(e) => handleParameterChange(paramName, e.target.value)}
                            fullWidth
                            margin="normal"
                        />
                    ))}
                </DialogContent>
                <DialogActions>
                    <Button onClick={handleCloseDialog}>Done</Button>
                </DialogActions>
            </Dialog>
        </Box>
    );
};

export default FeatureComposer;