import numpy as np
import pandas as pd
import os

def generate_sensor_dataset(n_samples=10000, random_state=42):
    np.random.seed(random_state)
    
    # 1. Base healthy operating conditions
    # ADXL345 Accelerometer (in g or m/s^2, 1g = 9.81 m/s^2)
    vib_x = np.random.normal(loc=0.05, scale=0.3, size=n_samples)
    vib_y = np.random.normal(loc=-0.02, scale=0.3, size=n_samples)
    vib_z = np.random.normal(loc=1.0, scale=0.2, size=n_samples) # ~1g gravity baseline
    
    # DS18B20 Temperature (°C)
    temperature = np.random.normal(loc=42.0, scale=6.0, size=n_samples)
    
    # Voltage Sensor (Nominal 12.0 V DC)
    voltage = np.random.normal(loc=12.0, scale=0.35, size=n_samples)
    
    # Target failure flags
    vaf = np.zeros(n_samples, dtype=int) # Vibration Anomaly Failure (Bearing/Mechanical)
    ohf = np.zeros(n_samples, dtype=int) # Overheating Failure
    vlf = np.zeros(n_samples, dtype=int) # Voltage Fault (Under/Over Voltage)
    csf = np.zeros(n_samples, dtype=int) # Combined Stress Failure
    
    # 2. Inject realistic failure scenarios (~3.5 - 4% failure rate)
    
    # Scenario A: Vibration Anomaly Failure (VAF) ~ 120 samples
    vaf_indices = np.random.choice(n_samples, size=120, replace=False)
    vib_x[vaf_indices] += np.random.uniform(2.5, 6.0, size=120) * np.random.choice([-1, 1], size=120)
    vib_y[vaf_indices] += np.random.uniform(2.5, 6.0, size=120) * np.random.choice([-1, 1], size=120)
    vib_z[vaf_indices] += np.random.uniform(3.0, 7.0, size=120)
    vaf[vaf_indices] = 1
    
    # Scenario B: Overheating Failure (OHF) ~ 110 samples
    ohf_indices = np.random.choice(n_samples, size=110, replace=False)
    temperature[ohf_indices] += np.random.uniform(32.0, 50.0, size=110)
    ohf[ohf_indices] = 1
    
    # Scenario C: Voltage Fault Failure (VLF) ~ 100 samples
    vlf_indices = np.random.choice(n_samples, size=100, replace=False)
    # Half under-voltage, half over-voltage/surge
    vlf_under = vlf_indices[:50]
    vlf_over = vlf_indices[50:]
    voltage[vlf_under] -= np.random.uniform(2.2, 4.5, size=50) # Drops below 9.8V
    voltage[vlf_over] += np.random.uniform(2.5, 5.0, size=50)  # Surges above 14.5V
    vlf[vlf_indices] = 1
    
    # Scenario D: Combined Stress Failure (CSF) ~ 50 samples
    csf_indices = np.random.choice(n_samples, size=50, replace=False)
    vib_x[csf_indices] += np.random.uniform(1.8, 3.5, size=50)
    temperature[csf_indices] += np.random.uniform(20.0, 35.0, size=50)
    voltage[csf_indices] -= np.random.uniform(1.5, 3.0, size=50)
    csf[csf_indices] = 1
    
    # Overall Machine Failure Target
    machine_failure = ((vaf == 1) | (ohf == 1) | (vlf == 1) | (csf == 1)).astype(int)
    
    # Round numerical data to realistic sensor precision
    df = pd.DataFrame({
        'Vibration_x': np.round(vib_x, 3),
        'Vibration_y': np.round(vib_y, 3),
        'Vibration_z': np.round(vib_z, 3),
        'Temperature': np.round(temperature, 2),
        'Voltage': np.round(voltage, 2),
        'Machine_Failure': machine_failure,
        'VAF': vaf,
        'OHF': ohf,
        'VLF': vlf,
        'CSF': csf
    })
    
    return df

if __name__ == "__main__":
    df = generate_sensor_dataset()
    filepath = "predictive_maintenance.csv"
    df.to_csv(filepath, index=False)
    print(f"Generated new sensor dataset at {filepath}")
    print(f"Dataset Shape: {df.shape}")
    print(f"Failure Count:\n{df['Machine_Failure'].value_counts()}")
    print("\nFailure Mode Counts:")
    print(df[['VAF', 'OHF', 'VLF', 'CSF']].sum())
