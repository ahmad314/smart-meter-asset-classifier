import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis
import logging

logger = logging.getLogger(__name__)

class FeatureEngineer:
    """Handles transformation of raw 15-minute load profiles into engineered feature matrices."""
    
    def __init__(self):
        self.epsilon = 1e-6

    def generate_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extracts temporal markers from the timestamp."""
        df = df.copy()
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        df['hour_of_day'] = df['Timestamp'].dt.hour
        df['day_of_week'] = df['Timestamp'].dt.dayofweek
        df['month'] = df['Timestamp'].dt.month
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        df['date'] = df['Timestamp'].dt.date
        df['day_of_year'] = df['Timestamp'].dt.dayofyear
        df['week_of_year'] = df['Timestamp'].dt.isocalendar().week.astype(int)
        return df

    def compute_domain_signatures(self, series: pd.Series) -> dict:
        """Extracts targeted behavioral load signatures for a single meter."""
        features = {}
        
        # Diurnal behavior
        midday = series.between_time("12:00", "14:00").mean()
        features['midday_dip'] = midday - series.mean()
        
        eve = series.between_time("18:00", "21:00").mean()
        aft = series.between_time("15:00", "18:00").mean()
        features['evening_ramp'] = eve - aft
        
        night = series.between_time("00:00", "06:00").mean()
        day = series.between_time("06:00", "18:00").mean()
        features['night_day_ratio'] = night / (day + self.epsilon)
        
        wknd = series[series.index.dayofweek >= 5].mean()
        wkday = series[series.index.dayofweek < 5].mean()
        features['weekend_load_factor'] = wknd / (wkday + self.epsilon)
        
        # Volatility and spikes
        features['peak_to_avg'] = series.max() / (series.mean() + self.epsilon)
        
        threshold = series.mean()
        mask = series > threshold
        if mask.any():
            runs = (mask != mask.shift()).cumsum()
            features['longest_above_mean'] = int(mask.groupby(runs).sum().max())
        else:
            features['longest_above_mean'] = 0
            
        delta = series.diff()
        mask_inc = delta > 0
        if mask_inc.any():
            runs_inc = (mask_inc != mask_inc.shift()).cumsum()
            features['longest_increase_streak'] = int(mask_inc.groupby(runs_inc).sum().max())
        else:
            features['longest_increase_streak'] = 0
            
        return features

    def build_aggregate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculates daily aggregates and hourly profiles using vectorized operations."""
        logger.info("Computing daily aggregate features and lags...")
        
        # Lags
        df['consumption_lag_1'] = df.groupby('Meter_ID')['Consumption_kWh'].shift(1).fillna(0)
        df['injection_lag_1'] = df.groupby('Meter_ID')['Injection_kWh'].shift(1).fillna(0)
        df['consumption_lag_96'] = df.groupby('Meter_ID')['Consumption_kWh'].shift(96).fillna(0)
        df['injection_lag_96'] = df.groupby('Meter_ID')['Injection_kWh'].shift(96).fillna(0)

        # Daily aggregations
        daily_features = df.groupby(['Meter_ID', 'date']).agg(
            daily_avg_consumption=('Consumption_kWh', 'mean'),
            daily_max_consumption=('Consumption_kWh', 'max'),
            daily_std_consumption=('Consumption_kWh', 'std'),
            daily_avg_injection=('Injection_kWh', 'mean'),
            daily_max_injection=('Injection_kWh', 'max'),
            daily_std_injection=('Injection_kWh', 'std'),
            daily_consumption_injection_ratio=('Consumption_kWh', lambda x: x.sum() / (df.loc[x.index, 'Injection_kWh'].sum() + self.epsilon)),
            daily_net_consumption=('Consumption_kWh', lambda x: x.sum() - df.loc[x.index, 'Injection_kWh'].sum()),
            daily_skew_consumption=('Consumption_kWh', skew),
            daily_kurt_consumption=('Consumption_kWh', kurtosis),
            daily_peak_hour=('Consumption_kWh', lambda x: df.loc[x.index, 'hour_of_day'][x.idxmax()] if not x.empty else -1)
        ).reset_index()

        # Household level rollout
        num_cols = [c for c in daily_features.columns if c not in ['Meter_ID', 'date', 'daily_peak_hour'] and pd.api.types.is_numeric_dtype(daily_features[c])]
        household_features = daily_features.groupby('Meter_ID')[num_cols].mean().reset_index()
        
        # Peak hour mode
        mode_peak = daily_features.groupby('Meter_ID')['daily_peak_hour'].apply(lambda x: x.mode()[0] if not x.mode().empty else -1).reset_index(name='most_frequent_peak_hour')
        household_features = pd.merge(household_features, mode_peak, on='Meter_ID', how='left')
        household_features = pd.get_dummies(household_features, columns=['most_frequent_peak_hour'], prefix='peak_hour_mode')

        # Hourly profiles
        hourly_cons = df.groupby(['Meter_ID', 'hour_of_day'])['Consumption_kWh'].mean().unstack(fill_value=0)
        hourly_cons.columns = [f'avg_cons_hour_{h:02d}' for h in range(24)]
        hourly_inj = df.groupby(['Meter_ID', 'hour_of_day'])['Injection_kWh'].mean().unstack(fill_value=0)
        hourly_inj.columns = [f'avg_inj_hour_{h:02d}' for h in range(24)]

        household_features = household_features.merge(hourly_cons.reset_index(), on='Meter_ID', how='left')
        household_features = household_features.merge(hourly_inj.reset_index(), on='Meter_ID', how='left')

        return household_features

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Main pipeline sequence for feature extraction."""
        logger.info("Initializing feature engineering pipeline...")
        df = self.generate_time_features(df)
        
        # Domain signatures
        logger.info("Extracting custom behavioral signatures...")
        meters = df['Meter_ID'].unique()
        domain_features_list = []
        for m in meters:
            series = df[df['Meter_ID'] == m].set_index('Timestamp')['Consumption_kWh'].ffill().bfill()
            feats = self.compute_domain_signatures(series)
            feats['Meter_ID'] = m
            domain_features_list.append(feats)
            
        domain_df = pd.DataFrame(domain_features_list)
        
        # Aggregations
        agg_df = self.build_aggregate_features(df)
        
        # Merge all
        final_df = pd.merge(agg_df, domain_df, on='Meter_ID', how='left').fillna(0)
        logger.info(f"Feature engineering complete. Generated {final_df.shape[1] - 1} features.")
        return final_df
