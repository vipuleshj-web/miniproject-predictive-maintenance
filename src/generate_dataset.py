import numpy as np
import pandas as pd
import os

def generate_sensor_dataset(n_samples=750, random_state=42):
    """
    Generate synthetic sensor dataset aligned with ESP32 hardware:
    - temperature: DS18B20 (°C)
    - vibration: Instantaneous digital read (0 = steady/no vibration, 1 = vibration shock detected)
    - voltage: Raw 12-bit ADC reading (0 - 4095) from 0-25V voltage divider module
    - voltage_volts: Calculated voltage (ADC / 4095.0 * 25.0)
    """
    np.random.seed(random_state)
    
    # 1. Base healthy operating conditions (~80% healthy baseline)
    # DS18B20 Machine Temperature: healthy operating range around 38°C - 48°C
    temp = np.random.normal(loc=42.0, scale=4.0, size=n_samples)
    temp = np.clip(temp, 28.0, 52.0)
    
    # Digital Vibration sensor: in normal operation, 0 (no mechanical shock/trigger)
    # Small occasional false trigger rate ~1.5% due to environmental noise
    vib = np.random.choice([0, 1], size=n_samples, p=[0.985, 0.015])
    
    # Voltage sensor ADC (0-25V module, nominal 12.0V DC)
    # 12.0V / 25.0V * 4095 ≈ 1966 ADC counts
    voltage_adc = np.random.normal(loc=1966, scale=35, size=n_samples).astype(int)
    voltage_adc = np.clip(voltage_adc, 1850, 2080)
    
    # Fault targets
    vaf = np.zeros(n_samples, dtype=int)  # Vibration Anomaly Fault
    ohf = np.zeros(n_samples, dtype=int)  # Overheating Fault
    vlf = np.zeros(n_samples, dtype=int)  # Voltage Fault (Under/Over Voltage)
    csf = np.zeros(n_samples, dtype=int)  # Combined Stress Fault
    
    # 2. Inject Realistic Failure Scenarios (total ~150 failure samples)
    
    # Scenario A: Vibration Anomaly Fault (VAF) ~ 45 samples (Bearing wear, loose shaft, mechanical imbalance)
    vaf_indices = np.random.choice(n_samples, size=45, replace=False)
    vib[vaf_indices] = 1
    # Mechanical friction causes mild temperature rise
    temp[vaf_indices] += np.random.uniform(5.0, 12.0, size=45)
    vaf[vaf_indices] = 1
    
    # Scenario B: Overheating Fault (OHF) ~ 40 samples (Cooling failure, motor overload)
    available_indices = list(set(range(n_samples)) - set(vaf_indices))
    ohf_indices = np.random.choice(available_indices, size=40, replace=False)
    temp[ohf_indices] = np.random.uniform(72.0, 96.0, size=40)
    ohf[ohf_indices] = 1
    
    # Scenario C: Voltage Fault (VLF) ~ 40 samples (Supply drop/sag or voltage spike/surge)
    available_indices = list(set(available_indices) - set(ohf_indices))
    vlf_indices = np.random.choice(available_indices, size=40, replace=False)
    vlf_under = vlf_indices[:20]
    vlf_over = vlf_indices[20:]
    # Under-voltage: below 9.5V (< 1550 ADC counts)
    voltage_adc[vlf_under] = np.random.randint(1100, 1500, size=len(vlf_under))
    # Over-voltage: above 15.0V (> 2450 ADC counts)
    voltage_adc[vlf_over] = np.random.randint(2500, 3100, size=len(vlf_over))
    vlf[vlf_indices] = 1
    
    # Scenario D: Combined Stress Fault (CSF) ~ 25 samples (Heavy wear: vibration + heat + power strain)
    available_indices = list(set(available_indices) - set(vlf_indices))
    csf_indices = np.random.choice(available_indices, size=25, replace=False)
    vib[csf_indices] = 1
    temp[csf_indices] = np.random.uniform(65.0, 82.0, size=25)
    voltage_adc[csf_indices] = np.random.randint(1300, 1650, size=25)  # Under-voltage sag under load
    csf[csf_indices] = 1
    
    # Overall Machine Failure Target
    machine_failure = ((vaf == 1) | (ohf == 1) | (vlf == 1) | (csf == 1)).astype(int)
    
    # Calibrated voltage in Volts (0 - 25V scale)
    voltage_volts = np.round((voltage_adc / 4095.0) * 25.0, 2)
    
    df = pd.DataFrame({
        'temperature': np.round(temp, 2),
        'vibration': vib,
        'voltage': voltage_adc,
        'voltage_volts': voltage_volts,
        'Machine_Failure': machine_failure,
        'VAF': vaf,
        'OHF': ohf,
        'VLF': vlf,
        'CSF': csf
    })
    
    return df

if __name__ == "__main__":
    df = generate_sensor_dataset(n_samples=750, random_state=42)
    output_path = "predictive_maintenance.csv"
    df.to_csv(output_path, index=False)
    print(f"Dataset generated with {len(df)} samples at {output_path}")
    print("\nMachine Failure Distribution:")
    print(df['Machine_Failure'].value_counts())
    print("\nFailure Modes Breakdown:")
    print(df[['VAF', 'OHF', 'VLF', 'CSF']].sum())
    print("\nDataset Sample (First 5 rows):")
    print(df.head())
