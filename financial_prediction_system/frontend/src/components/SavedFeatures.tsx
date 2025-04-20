import React, { useState, useEffect } from 'react';
import {
    Paper,
    Typography,
    Table,
    TableBody,
    TableCell,
    TableContainer,
    TableHead,
    TableRow,
    IconButton,
    Button,
    Box,
    Dialog,
    DialogTitle,
    DialogContent,
    DialogActions,
    Alert
} from '@mui/material';
import DeleteIcon from '@mui/icons-material/Delete';
import EditIcon from '@mui/icons-material/Edit';
import PreviewIcon from '@mui/icons-material/Preview';
import { Editor } from '@monaco-editor/react';

interface Feature {
    id: number;
    name: string;
    formula: string;
    type: string;
    symbol: string;
    description?: string;
    mean?: number;
    std?: number;
    price_correlation?: number;
    returns_correlation?: number;
    created_at: string;
    updated_at: string;
}

interface SavedFeaturesProps {
    symbol: string;
    onPreview: (formula: string) => void;
    onEdit: (feature: Feature) => void;
}

const SavedFeatures: React.FC<SavedFeaturesProps> = ({
    symbol,
    onPreview,
    onEdit
}) => {
    const [features, setFeatures] = useState<Feature[]>([]);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [selectedFeature, setSelectedFeature] = useState<Feature | null>(null);
    const [dialogOpen, setDialogOpen] = useState(false);

    useEffect(() => {
        if (symbol) {
            fetchFeatures();
        }
    }, [symbol]);

    const fetchFeatures = async () => {
        setLoading(true);
        try {
            const response = await fetch(`/api/feature-engineering/features/${symbol}`);
            if (!response.ok) {
                throw new Error('Failed to fetch features');
            }
            const data = await response.json();
            setFeatures(data);
        } catch (err: any) {
            setError(err.message);
        } finally {
            setLoading(false);
        }
    };

    const handleDelete = async (id: number) => {
        try {
            const response = await fetch(`/api/feature-engineering/feature/${id}`, {
                method: 'DELETE'
            });
            if (!response.ok) {
                throw new Error('Failed to delete feature');
            }
            await fetchFeatures(); // Refresh the list
        } catch (err: any) {
            setError(err.message);
        }
    };

    const handlePreview = (feature: Feature) => {
        onPreview(feature.formula);
    };

    const handleFormulaView = (feature: Feature) => {
        setSelectedFeature(feature);
        setDialogOpen(true);
    };

    if (loading) {
        return (
            <Typography>Loading saved features...</Typography>
        );
    }

    if (error) {
        return (
            <Alert severity="error">{error}</Alert>
        );
    }

    return (
        <>
            <TableContainer component={Paper}>
                <Table>
                    <TableHead>
                        <TableRow>
                            <TableCell>Name</TableCell>
                            <TableCell>Type</TableCell>
                            <TableCell>Description</TableCell>
                            <TableCell>Price Correlation</TableCell>
                            <TableCell>Returns Correlation</TableCell>
                            <TableCell>Actions</TableCell>
                        </TableRow>
                    </TableHead>
                    <TableBody>
                        {features.map((feature) => (
                            <TableRow key={feature.id}>
                                <TableCell>{feature.name}</TableCell>
                                <TableCell>{feature.type}</TableCell>
                                <TableCell>{feature.description || '-'}</TableCell>
                                <TableCell>
                                    {feature.price_correlation?.toFixed(4) || '-'}
                                </TableCell>
                                <TableCell>
                                    {feature.returns_correlation?.toFixed(4) || '-'}
                                </TableCell>
                                <TableCell>
                                    <IconButton 
                                        onClick={() => handlePreview(feature)}
                                        title="Preview"
                                    >
                                        <PreviewIcon />
                                    </IconButton>
                                    <IconButton
                                        onClick={() => handleFormulaView(feature)}
                                        title="View Formula"
                                    >
                                        <EditIcon />
                                    </IconButton>
                                    <IconButton
                                        onClick={() => handleDelete(feature.id)}
                                        title="Delete"
                                    >
                                        <DeleteIcon />
                                    </IconButton>
                                </TableCell>
                            </TableRow>
                        ))}
                    </TableBody>
                </Table>
            </TableContainer>

            {/* Formula View Dialog */}
            <Dialog
                open={dialogOpen}
                onClose={() => setDialogOpen(false)}
                maxWidth="md"
                fullWidth
            >
                <DialogTitle>
                    Feature Formula: {selectedFeature?.name}
                </DialogTitle>
                <DialogContent>
                    <Editor
                        height="400px"
                        defaultLanguage="python"
                        value={selectedFeature?.formula || ''}
                        options={{
                            readOnly: true,
                            minimap: { enabled: false }
                        }}
                    />
                </DialogContent>
                <DialogActions>
                    <Button onClick={() => setDialogOpen(false)}>
                        Close
                    </Button>
                    <Button 
                        onClick={() => {
                            if (selectedFeature) {
                                onEdit(selectedFeature);
                            }
                            setDialogOpen(false);
                        }}
                        color="primary"
                    >
                        Edit Feature
                    </Button>
                </DialogActions>
            </Dialog>
        </>
    );
};

export default SavedFeatures;