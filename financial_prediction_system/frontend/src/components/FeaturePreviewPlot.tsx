import React from 'react';
import Plot from 'react-plotly.js';
import { Paper, Typography } from '@mui/material';

interface FeaturePreviewPlotProps {
    baseData: any;
    featureDefinition: string;
    targetDefinition?: string;
    featureData?: number[];
    targetData?: number[];
}

const FeaturePreviewPlot: React.FC<FeaturePreviewPlotProps> = ({
    baseData,
    featureDefinition,
    targetDefinition,
    featureData,
    targetData
}) => {
    if (!baseData || !featureDefinition || !featureData) {
        return (
            <Paper sx={{ p: 2, textAlign: 'center' }}>
                <Typography color="text.secondary">
                    Select data and define a feature to preview
                </Typography>
            </Paper>
        );
    }

    const plotData: any[] = [
        {
            x: baseData.date,
            y: featureData,
            type: 'scatter',
            mode: 'lines',
            name: 'Feature',
            yaxis: 'y'
        }
    ];

    // Add target data if available
    if (targetData && targetDefinition) {
        plotData.push({
            x: baseData.date,
            y: targetData,
            type: 'scatter',
            mode: 'lines',
            name: targetDefinition,
            yaxis: 'y2',
            line: { dash: 'dot' }
        });
    }

    const layout: Partial<Plotly.Layout> = {
        title: `Feature Preview${targetDefinition ? ' with Target' : ''}`,
        xaxis: { title: 'Date' },
        yaxis: { 
            title: 'Feature Value',
            side: 'left' as const
        },
        ...(targetData && targetDefinition ? {
            yaxis2: {
                title: 'Target Value',
                overlaying: 'y',
                side: 'right' as const
            }
        } : {}),
        showlegend: true,
        legend: {
            x: 0,
            y: 1.2,
            orientation: 'h' as const
        }
    };

    return (
        <Paper sx={{ p: 2 }}>
            <Plot
                data={plotData}
                layout={layout}
                style={{ width: '100%', height: '400px' }}
                config={{ responsive: true }}
            />
        </Paper>
    );
};

export default FeaturePreviewPlot;