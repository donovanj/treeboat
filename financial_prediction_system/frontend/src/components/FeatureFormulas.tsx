import React, { useState } from 'react';
import {
    Paper,
    Typography,
    TextField,
    Button,
    Box,
    Alert,
    Dialog,
    DialogTitle,
    DialogContent,
    DialogActions,
    List,
    ListItem,
    ListItemText,
    ListItemSecondaryAction,
    IconButton,
    Tooltip,
    Chip
} from '@mui/material';
import Editor from '@monaco-editor/react';
import EditIcon from '@mui/icons-material/Edit';
import DeleteIcon from '@mui/icons-material/Delete';
import InfoIcon from '@mui/icons-material/Info';

interface Formula {
    id: string;
    name: string;
    code: string;
    description?: string;
}

interface FeatureFormulasProps {
    onSaveFormula: (formula: Formula) => void;
    onDeleteFormula: (id: string) => void;
    savedFormulas: Formula[];
}

const defaultTemplate = `# Available variables:
# df - pandas DataFrame with columns: ['open', 'high', 'low', 'close', 'volume', 'date']
# np - numpy module
# pd - pandas module
# ta - TA-Lib module for technical analysis

# Example: 20-day moving average
result = ta.SMA(df['close'], timeperiod=20)

# Your feature calculation must assign the final result to 'result' variable`;

const FeatureFormulas: React.FC<FeatureFormulasProps> = ({
    onSaveFormula,
    onDeleteFormula,
    savedFormulas
}) => {
    const [isDialogOpen, setIsDialogOpen] = useState(false);
    const [currentFormula, setCurrentFormula] = useState<Formula | null>(null);
    const [formulaName, setFormulaName] = useState('');
    const [formulaDescription, setFormulaDescription] = useState('');
    const [formulaCode, setFormulaCode] = useState(defaultTemplate);
    const [error, setError] = useState<string | null>(null);

    const handleCreateNew = () => {
        setCurrentFormula(null);
        setFormulaName('');
        setFormulaDescription('');
        setFormulaCode(defaultTemplate);
        setIsDialogOpen(true);
        setError(null);
    };

    const handleEdit = (formula: Formula) => {
        setCurrentFormula(formula);
        setFormulaName(formula.name);
        setFormulaDescription(formula.description || '');
        setFormulaCode(formula.code);
        setIsDialogOpen(true);
        setError(null);
    };

    const validateFormula = (code: string): boolean => {
        // Basic validation - check if 'result' is assigned
        if (!code.includes('result =')) {
            setError('Formula must assign to "result" variable');
            return false;
        }
        return true;
    };

    const handleSave = () => {
        if (!formulaName.trim()) {
            setError('Formula name is required');
            return;
        }

        if (!validateFormula(formulaCode)) {
            return;
        }

        const formula: Formula = {
            id: currentFormula?.id || crypto.randomUUID(),
            name: formulaName,
            code: formulaCode,
            description: formulaDescription || undefined
        };

        onSaveFormula(formula);
        setIsDialogOpen(false);
    };

    return (
        <>
            <Paper sx={{ p: 2 }}>
                <Box display="flex" justifyContent="space-between" alignItems="center" mb={2}>
                    <Typography variant="h6">
                        Custom Feature Formulas
                    </Typography>
                    <Button
                        variant="contained"
                        onClick={handleCreateNew}
                    >
                        Create New Formula
                    </Button>
                </Box>

                <List>
                    {savedFormulas.map((formula) => (
                        <ListItem
                            key={formula.id}
                            divider
                            secondaryAction={
                                <Box>
                                    <IconButton
                                        edge="end"
                                        aria-label="edit"
                                        onClick={() => handleEdit(formula)}
                                        sx={{ mr: 1 }}
                                    >
                                        <EditIcon />
                                    </IconButton>
                                    <IconButton
                                        edge="end"
                                        aria-label="delete"
                                        onClick={() => onDeleteFormula(formula.id)}
                                    >
                                        <DeleteIcon />
                                    </IconButton>
                                </Box>
                            }
                        >
                            <ListItemText
                                primary={
                                    <Box display="flex" alignItems="center">
                                        {formula.name}
                                        {formula.description && (
                                            <Tooltip title={formula.description}>
                                                <InfoIcon
                                                    fontSize="small"
                                                    sx={{ ml: 1, opacity: 0.7 }}
                                                />
                                            </Tooltip>
                                        )}
                                    </Box>
                                }
                                secondary={
                                    <Box mt={1}>
                                        <Chip
                                            label="Custom Formula"
                                            size="small"
                                            color="primary"
                                            variant="outlined"
                                        />
                                    </Box>
                                }
                            />
                        </ListItem>
                    ))}
                </List>
            </Paper>

            <Dialog
                open={isDialogOpen}
                onClose={() => setIsDialogOpen(false)}
                maxWidth="md"
                fullWidth
            >
                <DialogTitle>
                    {currentFormula ? 'Edit Formula' : 'Create New Formula'}
                </DialogTitle>
                <DialogContent>
                    <Box sx={{ mt: 2 }}>
                        <TextField
                            label="Formula Name"
                            value={formulaName}
                            onChange={(e) => setFormulaName(e.target.value)}
                            fullWidth
                            margin="normal"
                            required
                        />
                        <TextField
                            label="Description (optional)"
                            value={formulaDescription}
                            onChange={(e) => setFormulaDescription(e.target.value)}
                            fullWidth
                            margin="normal"
                            multiline
                            rows={2}
                        />
                        <Box sx={{ mt: 2, mb: 2 }}>
                            <Typography variant="subtitle2" gutterBottom>
                                Python Code
                            </Typography>
                            <Editor
                                height="400px"
                                defaultLanguage="python"
                                value={formulaCode}
                                onChange={(value) => setFormulaCode(value || '')}
                                options={{
                                    minimap: { enabled: false },
                                    scrollBeyondLastLine: false,
                                    fontSize: 14
                                }}
                            />
                        </Box>
                        {error && (
                            <Alert severity="error" sx={{ mt: 2 }}>
                                {error}
                            </Alert>
                        )}
                    </Box>
                </DialogContent>
                <DialogActions>
                    <Button onClick={() => setIsDialogOpen(false)}>Cancel</Button>
                    <Button onClick={handleSave} variant="contained">
                        Save
                    </Button>
                </DialogActions>
            </Dialog>
        </>
    );
};

export default FeatureFormulas;