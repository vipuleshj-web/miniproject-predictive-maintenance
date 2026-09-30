import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os

FEATURE_COLUMNS = [
    'Vibration_x',
    'Vibration_y',
    'Vibration_z',
    'Temperature',
    'Voltage',
    'Vibration_RMS',
    'Voltage_Deviation',
    'Power_Factor_Proxy',
    'Temp_Anomaly'
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
    Perform physics-informed feature engineering for ADXL345, DS18B20, and Voltage Sensor:
    - Vibration_RMS: sqrt(Vx^2 + Vy^2 + Vz^2) total vibration acceleration vector magnitude
    - Voltage_Deviation: abs(Voltage - 12.0) deviation from nominal 12V DC power supply
    - Power_Factor_Proxy: Voltage * Vibration_RMS electromechanical dynamic stress index
    - Temp_Anomaly: Temperature - 25.0 (rise over 25°C standard ambient)
    """
    df_feat = df.copy()
    
    # 1. Total Vibration Vector Magnitude (RMS in g units)
    df_feat['Vibration_RMS'] = np.sqrt(
        df_feat['Vibration_x']**2 + df_feat['Vibration_y']**2 + df_feat['Vibration_z']**2
    )
    
    # 2. Voltage Supply Instability Deviation (from 12.0 V nominal)
    df_feat['Voltage_Deviation'] = np.abs(df_feat['Voltage'] - 12.0)
    
    # 3. Electromechanical Dynamic Stress Index
    df_feat['Power_Factor_Proxy'] = df_feat['Voltage'] * df_feat['Vibration_RMS']
    
    # 4. Thermal Stress Delta (above 25°C ambient)
    df_feat['Temp_Anomaly'] = df_feat['Temperature'] - 25.0
    
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
    print("Data loading & preprocessing successful!")
    print(f"X_train shape: {data_dict['X_train'].shape}")
    print(f"X_test shape: {data_dict['X_test'].shape}")
