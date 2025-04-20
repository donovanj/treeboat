import React, { useState, useEffect } from 'react';
import { 
    Paper, 
    Typography, 
    Select, 
    MenuItem, 
    TextField, 
    Button,
    FormControl,
    InputLabel,
    Box,
    Chip,
    Grid
} from '@mui/material';
import Editor from '@monaco-editor/react';

interface FeatureBuilderProps {
    onPreview: (formula: string) => void;
    onSave: (name: string, formula: string, type: string) => void;
    predefinedFeatures?: string[];
    initialFeature?: {
        name: string;
        formula: string;
        type: string;
    };
}

type TransformationType = 'Rolling Window' | 'Gap Analysis' | 'Volume Profile' | 'Custom Formula';

const TRANSFORMATION_TEMPLATES: Record<TransformationType, string> = {
    'Rolling Window': `# Available variables: df (DataFrame with OHLCV data)
# Example: 20-day moving average of close price
df['close'].rolling(window=20).mean()`,
    'Gap Analysis': `# Calculate gap features
gap = (df['open'] - df['close'].shift(1)) / df['close'].shift(1)
gap_volume = gap * df['volume'] / df['volume'].rolling(20).mean()
# Return the feature
gap_volume`,
    'Volume Profile': `# Volume-based feature
vol_ma = df['volume'].rolling(window=20).mean()
price_vol = df['close'] * df['volume'] / vol_ma
# Return the feature
price_vol`,
    'Custom Formula': `# Write your custom feature calculation here
# Available variables: df (DataFrame with OHLCV data)
# Example: Custom momentum indicator
momentum = df['close'] / df['close'].shift(5) - 1
# Return the feature
momentum`
};

const FeatureBuilder: React.FC<FeatureBuilderProps> = ({
    onPreview,
    onSave,
    predefinedFeatures = [],
    initialFeature
}) => {
    const [transformation, setTransformation] = useState<TransformationType>(
        initialFeature?.type as TransformationType || 'Custom Formula'
    );
    const [formula, setFormula] = useState(
        initialFeature?.formula || TRANSFORMATION_TEMPLATES['Custom Formula']
    );
    const [featureName, setFeatureName] = useState(initialFeature?.name || '');
    const [selectedFeatures, setSelectedFeatures] = useState<string[]>([]);

    useEffect(() => {
        if (initialFeature) {
            setTransformation(initialFeature.type as TransformationType);
            setFormula(initialFeature.formula);
            setFeatureName(initialFeature.name);
        }
    }, [initialFeature]);

    const handleTransformationChange = (type: TransformationType) => {
        setTransformation(type);
        setFormula(TRANSFORMATION_TEMPLATES[type]);
    };

    const handleFeatureClick = (feature: string) => {
        const newFeatures = selectedFeatures.includes(feature)
            ? selectedFeatures.filter(f => f !== feature)
            : [...selectedFeatures, feature];
        setSelectedFeatures(newFeatures);
        
        // Update formula to include selected features
        if (transformation === 'Custom Formula') {
            const featureVars = newFeatures.map(f => `${f} = df['${f}']`).join('\n');
            setFormula(`# Selected features:\n${featureVars}\n\n# Your calculation here:\nresult = # use the features above\n# Return the feature\nresult`);
        }
    };

    return (
        <Paper sx={{ p: 2 }}>
            <Typography variant="h6" gutterBottom>
                Feature Builder
            </Typography>

            <Grid container spacing={2}>
                {/* Transformation Type Selector */}
                <Grid item xs={12}>
                    <FormControl fullWidth>
                        <InputLabel>Transformation Type</InputLabel>
                        <Select
                            value={transformation}
                            label="Transformation Type"
                            onChange={(e) => handleTransformationChange(e.target.value as TransformationType)}
                        >
                            {Object.keys(TRANSFORMATION_TEMPLATES).map(type => (
                                <MenuItem key={type} value={type}>{type}</MenuItem>
                            ))}
                        </Select>
                    </FormControl>
                </Grid>

                {/* Available Features */}
                <Grid item xs={12}>
                    <Typography variant="subtitle2" gutterBottom>
                        Available Features
                    </Typography>
                    <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1 }}>
                        {predefinedFeatures.map((feature) => (
                            <Chip
                                key={feature}
                                label={feature}
                                onClick={() => handleFeatureClick(feature)}
                                color={selectedFeatures.includes(feature) ? "primary" : "default"}
                            />
                        ))}
                    </Box>
                </Grid>

                {/* Code Editor */}
                <Grid item xs={12}>
                    <Editor
                        height="300px"
                        defaultLanguage="python"
                        value={formula}
                        onChange={(value) => setFormula(value || '')}
                        theme="vs-dark"
                        options={{
                            minimap: { enabled: false },
                            fontSize: 14,
                            lineNumbers: 'on',
                            roundedSelection: false,
                            scrollBeyondLastLine: false,
                            readOnly: false,
                            automaticLayout: true,
                        }}
                    />
                </Grid>

                {/* Feature Name Input */}
                <Grid item xs={12}>
                    <TextField
                        fullWidth
                        label="Feature Name"
                        value={featureName}
                        onChange={(e) => setFeatureName(e.target.value)}
                        sx={{ mb: 2 }}
                    />
                </Grid>

                {/* Action Buttons */}
                <Grid item xs={12}>
                    <Box sx={{ display: 'flex', gap: 2 }}>
                        <Button
                            variant="outlined"
                            onClick={() => onPreview(formula)}
                        >
                            Preview
                        </Button>
                        <Button
                            variant="contained"
                            onClick={() => onSave(featureName, formula, transformation)}
                            disabled={!featureName || !formula}
                        >
                            Save Feature
                        </Button>
                    </Box>
                </Grid>
            </Grid>
        </Paper>
    );
};

export default FeatureBuilder;