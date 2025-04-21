import React, { useEffect, useState } from 'react';
import {
    Accordion,
    AccordionSummary,
    AccordionDetails,
    Typography,
    List,
    ListItem,
    ListItemText,
    ListItemButton,
    Chip,
    Box,
    CircularProgress,
    Alert
} from '@mui/material';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';

export interface Parameter {
    name: string;
    type: string;
    description: string;
    defaultValue: any;
    range?: [number, number];
    options?: string[];
}

export interface Feature {
    name: string;
    description: string;
    parameters?: Parameter[];
}

export interface FeatureCategory {
    name: string;
    description: string;
    parameters: Parameter[];
}

interface FeatureCategoriesProps {
    onFeatureSelect: (feature: Feature) => void;
}

const FeatureCategories: React.FC<FeatureCategoriesProps> = ({ onFeatureSelect }) => {
    const [categories, setCategories] = useState<FeatureCategory[]>([]);
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);

    useEffect(() => {
        const fetchCategories = async () => {
            setIsLoading(true);
            try {
                const response = await fetch('/api/features/categories');
                if (!response.ok) {
                    throw new Error('Failed to fetch feature categories');
                }
                const data = await response.json();
                setCategories(data);
            } catch (err) {
                setError(err instanceof Error ? err.message : 'Failed to load categories');
            } finally {
                setIsLoading(false);
            }
        };

        fetchCategories();
    }, []);

    if (isLoading) {
        return (
            <Box display="flex" justifyContent="center" p={3}>
                <CircularProgress />
            </Box>
        );
    }

    if (error) {
        return (
            <Alert severity="error" sx={{ m: 2 }}>
                {error}
            </Alert>
        );
    }

    return (
        <div>
            {categories.map((category) => (
                <Accordion key={category.name}>
                    <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                        <Typography>{category.name}</Typography>
                    </AccordionSummary>
                    <AccordionDetails>
                        <Box>
                            <Typography variant="body2" color="text.secondary" gutterBottom>
                                {category.description}
                            </Typography>
                            <List>
                                <ListItem disablePadding>
                                    <ListItemButton onClick={() => onFeatureSelect({
                                        name: category.name,
                                        description: category.description,
                                        parameters: category.parameters
                                    })}>
                                        <ListItemText 
                                            primary="Add to Composition"
                                            secondary={
                                                <Box sx={{ mt: 1 }}>
                                                    {category.parameters.map(param => (
                                                        <Chip
                                                            key={param.name}
                                                            label={`${param.name}: ${param.defaultValue}`}
                                                            size="small"
                                                            sx={{ mr: 1, mb: 1 }}
                                                        />
                                                    ))}
                                                </Box>
                                            }
                                        />
                                    </ListItemButton>
                                </ListItem>
                            </List>
                        </Box>
                    </AccordionDetails>
                </Accordion>
            ))}
        </div>
    );
};

export default FeatureCategories;