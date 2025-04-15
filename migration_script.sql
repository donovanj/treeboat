-- Drop existing tables if they exist
DROP TABLE IF EXISTS ml_backtest_results CASCADE;
DROP TABLE IF EXISTS ml_feature_importance CASCADE;
DROP TABLE IF EXISTS ml_ensemble_model_mappings CASCADE;
DROP TABLE IF EXISTS ml_ensemble_models CASCADE;
DROP TABLE IF EXISTS ml_model_results CASCADE;

-- Create ml_model_results table with updated schema
CREATE TABLE ml_model_results (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR NOT NULL,
    model_type VARCHAR NOT NULL,
    target_type VARCHAR NOT NULL,
    feature_sets JSON NOT NULL,
    hyperparameters JSON,
    train_start_date TIMESTAMP NOT NULL,
    train_end_date TIMESTAMP NOT NULL,
    test_start_date TIMESTAMP,
    test_end_date TIMESTAMP,
    
    -- Model performance metrics
    test_mse FLOAT,
    test_r2 FLOAT,
    test_mae FLOAT,
    cv_mean_mse FLOAT,
    cv_std_mse FLOAT,
    cv_mean_r2 FLOAT,
    cv_std_r2 FLOAT,
    cv_mean_mae FLOAT,
    cv_std_mae FLOAT,
    
    -- Model binary storage
    model_path VARCHAR,
    
    -- Metadata
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP NOT NULL DEFAULT NOW()
);

-- Create index on symbol
CREATE INDEX idx_ml_model_results_symbol ON ml_model_results(symbol);

-- Create feature_importance table
CREATE TABLE ml_feature_importance (
    id SERIAL PRIMARY KEY,
    model_result_id INTEGER NOT NULL,
    feature VARCHAR NOT NULL,
    importance FLOAT NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    
    CONSTRAINT fk_feature_importance_model
        FOREIGN KEY (model_result_id)
        REFERENCES ml_model_results (id)
        ON DELETE CASCADE
);

-- Create ensemble_models table
CREATE TABLE ml_ensemble_models (
    id SERIAL PRIMARY KEY,
    name VARCHAR NOT NULL,
    ensemble_method VARCHAR NOT NULL,
    hyperparameters JSON,
    model_path VARCHAR,
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP NOT NULL DEFAULT NOW()
);

-- Create ensemble_model_mappings table
CREATE TABLE ml_ensemble_model_mappings (
    ensemble_id INTEGER NOT NULL,
    model_id INTEGER NOT NULL,
    weight FLOAT,
    
    PRIMARY KEY (ensemble_id, model_id),
    
    CONSTRAINT fk_ensemble_mapping_ensemble
        FOREIGN KEY (ensemble_id)
        REFERENCES ml_ensemble_models (id)
        ON DELETE CASCADE,
        
    CONSTRAINT fk_ensemble_mapping_model
        FOREIGN KEY (model_id)
        REFERENCES ml_model_results (id)
        ON DELETE CASCADE
);

-- Create backtest_results table with updated schema
CREATE TABLE ml_backtest_results (
    id SERIAL PRIMARY KEY,
    model_id INTEGER,
    ensemble_id INTEGER,
    symbol VARCHAR NOT NULL,
    start_date TIMESTAMP NOT NULL,
    end_date TIMESTAMP NOT NULL,
    performance_metrics JSON NOT NULL,
    selected_features JSON,
    parameters JSON,
    trades JSON,
    equity_curve JSON,
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP NOT NULL DEFAULT NOW(),
    
    CONSTRAINT fk_backtest_model
        FOREIGN KEY (model_id)
        REFERENCES ml_model_results (id)
        ON DELETE SET NULL,
        
    CONSTRAINT fk_backtest_ensemble
        FOREIGN KEY (ensemble_id)
        REFERENCES ml_ensemble_models (id)
        ON DELETE SET NULL,
        
    CONSTRAINT check_model_or_ensemble
        CHECK ((model_id IS NULL AND ensemble_id IS NOT NULL) OR
               (model_id IS NOT NULL AND ensemble_id IS NULL))
);

-- Create index on symbol for backtest results
CREATE INDEX idx_ml_backtest_results_symbol ON ml_backtest_results(symbol);

-- Create a function to update the updated_at timestamp
CREATE OR REPLACE FUNCTION update_modified_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Create triggers for updated_at fields
CREATE TRIGGER update_ml_model_results_modtime
    BEFORE UPDATE ON ml_model_results
    FOR EACH ROW
    EXECUTE FUNCTION update_modified_column();
    
CREATE TRIGGER update_ml_ensemble_models_modtime
    BEFORE UPDATE ON ml_ensemble_models
    FOR EACH ROW
    EXECUTE FUNCTION update_modified_column();
    
CREATE TRIGGER update_ml_backtest_results_modtime
    BEFORE UPDATE ON ml_backtest_results
    FOR EACH ROW
    EXECUTE FUNCTION update_modified_column();