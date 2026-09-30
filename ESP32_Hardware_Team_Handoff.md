# 🚀 ESP32 Predictive Maintenance Firmware & Hardware Deployment Guide

## 📌 Overview
This document contains the complete firmware, wiring setup, library dependencies, and upload instructions for the ESP32 Predictive Maintenance sensor node.

The ESP32 collects live readings from:
1. **ADXL345** (3-axis vibration/acceleration)
2. **DS18B20** (Machine surface temperature)
3. **Analog Voltage Sensor** (12V DC power supply monitoring)

The ESP32 processes sensor values and streams JSON formatted sensor data every 1 second over **Serial UART (115200 Baud)**.

---

## 🔌 1. Hardware Pinout & Wiring Diagram

| Sensor Module | Sensor Pin | ESP32 Pin | Wiring / Connection Notes |
| :--- | :--- | :--- | :--- |
| **ADXL345 (I2C)** | `VCC` | `3.3V` or `5V` | Power supply |
| | `GND` | `GND` | Common Ground |
| | `SDA` | **`GPIO 21`** | I2C Data Line |
| | `SCL` | **`GPIO 22`** | I2C Clock Line |
| **DS18B20 (Temp)** | `VCC` | `3.3V` or `5V` | Power supply |
| | `GND` | `GND` | Common Ground |
| | `DATA` | **`GPIO 4`** | **CRITICAL:** Requires a **4.7 kΩ pull-up resistor** connected between DATA pin and VCC! |
| **Voltage Sensor** | `VCC` / `+` | Motor / DC Line `+` | Measures up to 25V DC |
| | `GND` / `-` | DC Line `-` / `GND` | Common Ground |
| | `S` (Signal Out) | **`GPIO 34`** | ESP32 ADC Pin |

---

## 📚 2. Required Arduino IDE Libraries
Please install the following libraries in Arduino IDE (**Sketch** -> **Include Library** -> **Manage Libraries...**):

1. `Adafruit ADXL345` (by Adafruit)
2. `Adafruit Unified Sensor` (by Adafruit)
3. `DallasTemperature` (by Miles Burton)
4. `OneWire` (by Paul Stoffregen)
5. `ArduinoJson` (by Benoit Blanchon — version 6 or 7)

---

## 💻 3. ESP32 Arduino C++ Code (`esp32_predictive_maintenance.ino`)

Save and compile the following code in Arduino IDE:

```cpp
#include <Wire.h>
#include <Adafruit_Sensor.h>
#include <Adafruit_ADXL345_U.h>
#include <OneWire.h>
#include <DallasTemperature.h>
#include <ArduinoJson.h>

// --- PIN DEFINITIONS ---
#define ONE_WIRE_BUS 4        // DS18B20 Data Pin (GPIO 4)
#define VOLTAGE_SENSOR_PIN 34 // Voltage Sensor Signal Pin (GPIO 34 ADC)

// Voltage Sensor Divider Constants (Standard 25V DC Voltage Divider Module 5:1 ratio)
#define ADC_REF_VOLTAGE 3.3
#define ADC_RESOLUTION 4095.0
#define VOLTAGE_DIVIDER_RATIO 5.0 // (R1 + R2) / R2 = (30K + 7.5K) / 7.5K = 5.0

// --- SENSOR INSTANCES ---
Adafruit_ADXL345_Unified accel = Adafruit_ADXL345_Unified(12345);
OneWire oneWire(ONE_WIRE_BUS);
DallasTemperature tempSensor(&oneWire);

void setup() {
  Serial.begin(115200);
  while (!Serial && millis() < 3000);

  Serial.println("Initializing ESP32 Predictive Maintenance Sensor Suite...");

  // 1. ADXL345 Accelerometer
  if (!accel.begin()) {
    Serial.println("ERROR: Could not find ADXL345! Check SDA/SCL wiring.");
    while (1);
  }
  accel.setRange(ADXL345_RANGE_16_G);

  // 2. DS18B20 Temperature Sensor
  tempSensor.begin();

  // 3. Voltage Sensor Pin
  analogReadResolution(12);
  pinMode(VOLTAGE_SENSOR_PIN, INPUT);

  Serial.println("Sensors Initialized & Ready!");
  delay(1000);
}

void loop() {
  // --- READ ADXL345 VIBRATION ---
  sensors_event_t event;
  accel.getEvent(&event);
  float vib_x = event.acceleration.x / 9.80665;
  float vib_y = event.acceleration.y / 9.80665;
  float vib_z = event.acceleration.z / 9.80665;

  // --- READ DS18B20 TEMPERATURE ---
  tempSensor.requestTemperatures();
  float temperature = tempSensor.getTempCByIndex(0);
  if (temperature == DEVICE_DISCONNECTED_C) {
    temperature = 25.0; // Fallback value if disconnected
  }

  // --- READ VOLTAGE SENSOR ---
  int rawADC = analogRead(VOLTAGE_SENSOR_PIN);
  float pinVoltage = (rawADC / ADC_RESOLUTION) * ADC_REF_VOLTAGE;
  float actualVoltage = pinVoltage * VOLTAGE_DIVIDER_RATIO;

  // --- SERIAL JSON TRANSMISSION ---
  StaticJsonDocument<256> doc;
  doc["Vibration_x"] = round(vib_x * 1000.0) / 1000.0;
  doc["Vibration_y"] = round(vib_y * 1000.0) / 1000.0;
  doc["Vibration_z"] = round(vib_z * 1000.0) / 1000.0;
  doc["Temperature"] = round(temperature * 100.0) / 100.0;
  doc["Voltage"]     = round(actualVoltage * 100.0) / 100.0;

  serializeJson(doc, Serial);
  Serial.println();

  delay(1000); // 1 Second interval
}
```

---

## ⚡ 4. Upload & Verification Steps for Hardware Team

1. Connect ESP32 via Micro-USB / USB-C data cable to PC.
2. Select Board: **Tools -> Board -> ESP32 Arduino -> ESP32 Dev Module**
3. Select COM Port: **Tools -> Port -> COMx**
4. Click **Upload**. *(If flashing hangs at `Connecting........_____`, press and hold the **BOOT** button on ESP32 until writing starts).*
5. Open **Serial Monitor** (**Tools -> Serial Monitor**) and set Baud Rate to **`115200`**.
6. **Expected Output Verification**:
   The Serial Monitor should display continuous valid JSON lines like this:
   ```json
   {"Vibration_x":0.045,"Vibration_y":-0.018,"Vibration_z":0.998,"Temperature":34.25,"Voltage":12.04}
   ```
