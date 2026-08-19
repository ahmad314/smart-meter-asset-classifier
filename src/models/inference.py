import joblib
import pandas as pd
import logging
from src.features.build_features import FeatureEngineer

logger = logging.getLogger(__name__)

class InferencePipeline:
    def __init__(self, model_path: str, scaler_path: str, feature_columns: list):
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        self.feature_columns = feature_columns
        self.engineer = FeatureEngineer()

    def predict(self, raw_df: pd.DataFrame) -> pd.Series:
        """Executes the end-to-end inference pass for incoming raw data."""
        logger.info("Extracting features for incoming payload...")
        features_df = self.engineer.fit_transform(raw_df)
        
        # Enforce column schema matching the training environment
        X = features_df.reindex(columns=self.feature_columns, fill_value=0).fillna(0)
        
        logger.info("Scaling features and generating predictions...")
        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled)
        
        return pd.Series(predictions, index=features_df['Meter_ID'], name="Predicted_Label")
