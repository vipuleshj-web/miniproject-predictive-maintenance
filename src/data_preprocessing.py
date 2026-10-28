import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os

FEATURE_COLUMNS = [
    'temperature',
    'vibration',
    'voltage',
    'voltage_volts',
    'voltage_dev',
    'temp_anomaly',
    'stress_index'
]

TARGET_COLUMN = 'Machine_Failure'
FAILURE_MODES = ['VAF', 'OHF', 'VLF', 'CSF']

def load_data(filepath="predictive_maintenance.csv"):
    """Load the predictive maintenance sensor dataset."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset file not found at: {filepath}")
    df = pd.read_csv(filepath)
    return df

def engineer_features(df):
    """
    Perform physics-informed feature engineering for ESP32 sensor suite:
    - temperature: DS18B20 (°C)
    - vibration: Instantaneous digital read (0 or 1)
    - voltage: Raw 12-bit ADC (0 - 4095)
    
    Engineered Features:
    - voltage_volts: Derived from 0-25V voltage divider: (ADC / 4095.0) * 25.0
    - voltage_dev: abs(voltage_volts - 12.0) deviation from nominal 12V DC power
    - temp_anomaly: temperature - 25.0 (thermal rise above 25°C ambient)
    - stress_index: composite stress index combining mechanical shock, thermal elevation, and power strain
    """
    df_feat = df.copy()
    
    # Impute missing/null temperature (e.g. if DS18B20 was disconnected)
    if 'temperature' in df_feat.columns:
        df_feat['temperature'] = pd.to_numeric(df_feat['temperature'], errors='coerce').fillna(25.0)
    
    # Ensure vibration is numeric 0 or 1
    if 'vibration' in df_feat.columns:
        df_feat['vibration'] = pd.to_numeric(df_feat['vibration'], errors='coerce').fillna(0).astype(int)
        
    # Ensure voltage ADC is numeric
    if 'voltage' in df_feat.columns:
        df_feat['voltage'] = pd.to_numeric(df_feat['voltage'], errors='coerce').fillna(1966)
        
    # 1. Calibrated voltage in Volts (0-25V scale)
    if 'voltage_volts' not in df_feat.columns:
        df_feat['voltage_volts'] = (df_feat['voltage'] / 4095.0) * 25.0
    df_feat['voltage_volts'] = np.round(df_feat['voltage_volts'], 2)
    
    # 2. Voltage Supply Instability Deviation (from 12.0V nominal)
    df_feat['voltage_dev'] = np.round(np.abs(df_feat['voltage_volts'] - 12.0), 3)
    
    # 3. Thermal Stress Delta (rise above 25°C ambient)
    df_feat['temp_anomaly'] = np.round(df_feat['temperature'] - 25.0, 2)
    
    # 4. Composite Electromechanical Stress Index
    temp_excess = np.maximum(0.0, df_feat['temperature'] - 50.0)
    df_feat['stress_index'] = np.round(
        (df_feat['vibration'] * 2.5) + (df_feat['voltage_dev'] * 1.5) + (temp_excess * 0.1),
        3
    )
    
    return df_feat

def prepare_train_test_data(filepath="predictive_maintenance.csv", test_size=0.2, random_state=42):
    """
    Load data, engineer features, scale numerical features, and split into train/test sets.
    """
    raw_df = load_data(filepath)
    df_processed = engineer_features(raw_df)
    
    X = df_processed[FEATURE_COLUMNS]
    y = df_processed[TARGET_COLUMN]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=FEATURE_COLUMNS, index=X_train.index)
    X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=FEATURE_COLUMNS, index=X_test.index)
    
    return {
        'X_train': X_train_scaled_df,
        'X_test': X_test_scaled_df,
        'X_train_raw': X_train,
        'X_test_raw': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'scaler': scaler,
        'feature_names': FEATURE_COLUMNS,
        'df_processed': df_processed
    }

if __name__ == "__main__":
    data_dict = prepare_train_test_data()
    print("ESP32 Data loading & feature engineering successful!")
    print(f"Features: {FEATURE_COLUMNS}")
    print(f"X_train shape: {data_dict['X_train'].shape}")
    print(f"X_test shape: {data_dict['X_test'].shape}")
