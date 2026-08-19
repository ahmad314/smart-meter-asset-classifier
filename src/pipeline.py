import os
import logging
import pandas as pd
from src.features.build_features import FeatureEngineer
from src.models.train import ModelPipeline

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_training_pipeline(parquet_path: str, meta_path: str):
    """Orchestrates data loading, feature engineering, and model training."""
    
    logger.info("Loading telemetry and metadata...")
    df_smart_meter = pd.read_parquet(parquet_path)
    if df_smart_meter.index.name == 'Meter_ID':
        df_smart_meter = df_smart_meter.reset_index()
        
    df_meta = pd.read_csv(meta_path)
    df_combined = pd.merge(df_smart_meter, df_meta, on='Meter_ID', how='inner')
    
    # Feature Engineering Step
    engineer = FeatureEngineer()
    feature_matrix = engineer.fit_transform(df_combined)
    
    # Re-attach labels safely
    final_df = pd.merge(feature_matrix, df_meta[['Meter_ID', 'Category_Label']], on='Meter_ID', how='inner')
    X = final_df.drop(columns=['Meter_ID', 'Category_Label'])
    y = final_df['Category_Label']
    
    # Model Training Step
    trainer = ModelPipeline(output_dir='outputs')
    results = trainer.train_and_evaluate(X, y)
    
    logger.info("Pipeline execution finished successfully.")
    
    # Save the expected feature schema for inference downstream
    pd.Series(X.columns).to_csv('outputs/models/feature_schema.csv', index=False)
    
if __name__ == '__main__':
    # Defaulting to sample paths for ease of testing by reviewers
    run_training_pipeline(
        parquet_path='data/sample/smart_meter_data_sample.parquet',
        meta_path='data/sample/metadata_sample.csv'
    )
