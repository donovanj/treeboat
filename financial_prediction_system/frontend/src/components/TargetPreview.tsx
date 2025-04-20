import React from 'react';
import {
    Paper,
    Typography,
    FormControl,
    InputLabel,
    Select,
    MenuItem,
    TextField,
    Box
} from '@mui/material';
import Plot from 'react-plotly.js';

interface TargetPreviewProps {
    baseData: any;
    featureValues: number[];
    onTargetChange: (targetType: string, params: any) => void;
}

interface TargetDefinition {
    name: string;
    description: string;
    parameters: {
        name: string;
        type: 'number' | 'string';
        default: any;
        description: string;
    }[];
}

const TARGET_DEFINITIONS: TargetDefinition[] = [
    {
        name: 'forward_return',
        description: 'Simple n-day forward return',
        parameters: [
            {
                name: 'horizon',
                type: 'number',
                default: 5,
                description: 'Number of days to look forward'
            }
        ]
    },
    {
        name: 'binary_direction',
        description: 'Binary classification of price direction',
        parameters: [
            {
                name: 'horizon',
                type: 'number',
                default: 5,
                description: 'Number of days to look forward'
            },
            {
                name: 'threshold',
                type: 'number',
                default: 0,
                description: 'Return threshold for positive class'
            }
        ]
    },
    {
        name: 'volatility_adjusted_return',
        description: 'Forward return normalized by recent volatility',
        parameters: [
            {
                name: 'horizon',
                type: 'number',
                default: 5,
                description: 'Number of days to look forward'
            },
            {
                name: 'vol_window',
                type: 'number',
                default: 21,
                description: 'Window for volatility calculation'
            }
        ]
    }
];

const TargetPreview: React.FC<TargetPreviewProps> = ({
    baseData,
    featureValues,
    onTargetChange
}) => {
    const [selectedTarget, setSelectedTarget] = React.useState(TARGET_DEFINITIONS[0].name);
    const [parameters, setParameters] = React.useState<Record<string, any>>({
        horizon: TARGET_DEFINITIONS[0].parameters[0].default
    });

    const handleTargetChange = (event: any) => {
        const targetName = event.target.value;
        setSelectedTarget(targetName);
        
        // Reset parameters to defaults for new target
        const newTarget = TARGET_DEFINITIONS.find(t => t.name === targetName);
        if (newTarget) {
            const defaultParams = Object.fromEntries(
                newTarget.parameters.map(p => [p.name, p.default])
            );
            setParameters(defaultParams);
            onTargetChange(targetName, defaultParams);
        }
    };

    const handleParameterChange = (paramName: string, value: any) => {
        const newParams = { ...parameters, [paramName]: value };
        setParameters(newParams);
        onTargetChange(selectedTarget, newParams);
    };

    const selectedDefinition = TARGET_DEFINITIONS.find(t => t.name === selectedTarget);

    return (
        <Paper sx={{ p: 2 }}>
            <Typography variant="h6" gutterBottom>
                Target Preview
            </Typography>
            
            <FormControl fullWidth sx={{ mb: 2 }}>
                <InputLabel>Target Type</InputLabel>
                <Select
                    value={selectedTarget}
                    label="Target Type"
                    onChange={handleTargetChange}
                >
                    {TARGET_DEFINITIONS.map(target => (
                        <MenuItem key={target.name} value={target.name}>
                            {target.name}
                        </MenuItem>
                    ))}
                </Select>
            </FormControl>

            <Typography variant="body2" color="textSecondary" sx={{ mb: 2 }}>
                {selectedDefinition?.description}
            </Typography>

            <Box sx={{ mb: 2 }}>
                {selectedDefinition?.parameters.map(param => (
                    <TextField
                        key={param.name}
                        label={param.name}
                        type={param.type === 'number' ? 'number' : 'text'}
                        value={parameters[param.name]}
                        onChange={(e) => handleParameterChange(param.name, 
                            param.type === 'number' ? Number(e.target.value) : e.target.value)}
                        fullWidth
                        margin="normal"
                        helperText={param.description}
                    />
                ))}
            </Box>
        </Paper>
    );
};

export default TargetPreview;