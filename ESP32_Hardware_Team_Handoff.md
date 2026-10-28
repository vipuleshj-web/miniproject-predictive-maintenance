# 🚀 ESP32 Predictive Maintenance Firmware & ML Deployment Guide

## 📌 Overview
This document outlines the complete setup for connecting the ESP32 sensor node directly to the Machine Learning predictive maintenance pipeline.

The ESP32 collects live readings from:
1. **DS18B20** (Machine surface temperature in °C)
2. **Digital Vibration Sensor** (Instantaneous shock/vibration trigger on GPIO 27: `0` = normal, `1` = vibration detected)
3. **0-25V Voltage Divider Sensor** (12-bit ADC reading on GPIO 34, measuring nominal 12V DC)

The ESP32 streams JSON payloads every 3 seconds over Wi-Fi via HTTP POST to:
`http://<YOUR_LAPTOP_IP>:8000/sensor-data`

---

## 🔌 1. Hardware Pinout & Wiring Diagram

| Sensor Module | Sensor Pin | ESP32 Pin | Wiring / Connection Notes |
| :--- | :--- | :--- | :--- |
| **DS18B20 (Temp)** | `VCC` | `3.3V` or `5V` | Power supply |
| | `GND` | `GND` | Common Ground |
| | `DATA` | **`GPIO 4`** | **CRITICAL:** Requires a **4.7 kΩ pull-up resistor** connected between DATA pin and VCC! |
| **Digital Vibration Sensor** | `VCC` | `3.3V` or `5V` | Power supply |
| | `GND` | `GND` | Common Ground |
| | `DO` (Digital Out) | **`GPIO 27`** | Instantaneous digital signal (0 = quiet, 1 = shock) |
| **Voltage Sensor (0-25V)** | `VCC` / `+` | Machine DC Supply `+` | Measures up to 25V DC (Nominal ~12V DC) |
| | `GND` / `-` | Machine DC Supply `-` | Common Ground with ESP32 |
| | `S` (Signal Out) | **`GPIO 34`** | ESP32 ADC Input (Reads 0 to 4095) |
| **Status LED** | Onboard LED | **`GPIO 2`** | Automatically illuminates when ML backend detects `CRITICAL` fault |

---

## 💻 2. Running the Complete System

### Step 1: Find Your Laptop's Local Wi-Fi IPv4 Address
Open PowerShell and run:
```powershell
ipconfig
```
Look for **IPv4 Address** under your active Wi-Fi adapter (e.g. `10.129.160.169` or `192.168.1.x`).
Ensure this IP matches `API_URL` in [`esp32_firmware/esp32_firmware.ino`](file:///d:/miniproject/esp32_firmware/esp32_firmware.ino).

### Step 2: Start the Machine Learning Ingestion API Server
Open a terminal in the project directory and run:
```powershell
python api_server.py
```
This runs the FastAPI server on port `8000`. The server:
- Listens for `POST /sensor-data`
- Extracts physics-informed features (`stress_index`, `voltage_dev`, `temp_anomaly`, `voltage_volts`)
- Evaluates real-time failure risk using `best_model.joblib` (XGBoost)
- Diagnoses specific failure modes (`VAF`, `OHF`, `VLF`, `CSF`)
- Responds back to the ESP32 with health status and risk probability
- Logs telemetry to [`live_telemetry.json`](file:///d:/miniproject/live_telemetry.json)

### Step 3: Launch the Streamlit AI Dashboard
In a second terminal, run:
```powershell
streamlit run app.py
```
Open your browser to `http://localhost:8501`.
- **📡 Live ESP32 Telemetry Feed**: Real-time view of live sensor broadcasts from the ESP32.
- **🎛️ Hardware Telemetry Simulator**: Manual parameter tuning and fault scenario simulation.
- **📂 Batch Telemetry CSV Scoring**: Score batches of sensor logs or dataset files.
- **📊 Model Intelligence & Diagnostics**: Detailed model comparison, ROC-AUC, and feature importance.

---

## ⚡ 3. ESP32 Arduino C++ Code
The complete firmware sketch is located at:
[`esp32_firmware/esp32_firmware.ino`](file:///d:/miniproject/esp32_firmware/esp32_firmware.ino)
