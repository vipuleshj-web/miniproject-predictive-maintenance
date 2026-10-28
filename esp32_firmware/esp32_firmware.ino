#include <WiFi.h>
#include <HTTPClient.h>
#include <OneWire.h>
#include <DallasTemperature.h>

// ================= WIFI SETTINGS =================
const char* WIFI_SSID = "One Plus";
const char* WIFI_PASSWORD = "12345678";

// Replace with your laptop's CURRENT Wi-Fi IPv4 address.
// Note: Do not use localhost here; the ESP32 needs your laptop's local LAN IP.
const char* API_URL = "http://10.94.201.198:8000/sensor-data";

// ================= SENSOR PINS =================
#define VIBRATION_PIN 27      // Digital Vibration Sensor (SW-420 or ball-switch)
#define TEMPERATURE_PIN 4     // DS18B20 1-Wire Data pin (requires 4.7k pull-up to 3.3V)
#define VOLTAGE_PIN 34        // 0-25V Voltage Divider Sensor Signal (GPIO 34 ADC)
#define STATUS_LED_PIN 2      // Built-in ESP32 LED (GPIO 2) for visual alert indicator

// ================= TEMPERATURE SENSOR =================
OneWire oneWire(TEMPERATURE_PIN);
DallasTemperature temperatureSensor(&oneWire);

// ================= TIMING =================
unsigned long lastSendTime = 0;
const unsigned long SEND_INTERVAL = 3000; // Send telemetry every 3 seconds

// ================= WIFI CONNECTION =================
void connectWiFi() {
  Serial.print("Connecting to WiFi: ");
  Serial.println(WIFI_SSID);

  WiFi.mode(WL_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);

  int attempts = 0;
  while (WiFi.status() != WL_CONNECTED && attempts < 30) {
    delay(500);
    Serial.print(".");
    attempts++;
  }

  Serial.println();

  if (WiFi.status() == WL_CONNECTED) {
    Serial.println("✅ WiFi connected successfully!");
    Serial.print("ESP32 IP: ");
    Serial.println(WiFi.localIP());
  } else {
    Serial.println("❌ WiFi connection failed. Will retry on next send attempt.");
  }
}

// ================= SENSOR READINGS =================
float readTemperature() {
  temperatureSensor.requestTemperatures();
  float temp = temperatureSensor.getTempCByIndex(0);

  if (temp == DEVICE_DISCONNECTED_C || temp < -55.0 || temp > 125.0) {
    Serial.println("⚠️ Warning: DS18B20 Temperature sensor unavailable or disconnected.");
    return NAN;
  }
  return temp;
}

int readVibration() {
  // Instantaneous digital read: 0 = steady, 1 = shock/vibration trigger
  return digitalRead(VIBRATION_PIN);
}

int readVoltageADC() {
  // 12-bit ADC reading (0 to 4095)
  return analogRead(VOLTAGE_PIN);
}

// ================= SEND DATA & RECEIVE ML INFERENCE =================
void sendSensorData(float temperature, int vibration, int voltageADC) {
  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("WiFi disconnected. Reconnecting...");
    connectWiFi();

    if (WiFi.status() != WL_CONNECTED) {
      Serial.println("Skipping upload (WiFi unavailable).");
      return;
    }
  }

  HTTPClient http;
  http.setTimeout(5000);

  if (!http.begin(API_URL)) {
    Serial.println("Could not initialize HTTP connection to backend server.");
    return;
  }

  http.addHeader("Content-Type", "application/json");

  // Send null when the DS18B20 is unavailable so the backend can impute gracefully
  String temperatureJSON = isnan(temperature) ? "null" : String(temperature, 2);

  String payload = "{";
  payload += "\"temperature\":" + temperatureJSON + ",";
  payload += "\"vibration\":" + String(vibration) + ",";
  payload += "\"voltage\":" + String(voltageADC) + ",";
  payload += "\"current\":null,";
  payload += "\"power\":null,";
  payload += "\"rpm\":null";
  payload += "}";

  Serial.println("Uploading telemetry payload to ML backend:");
  Serial.println(payload);

  int httpCode = http.POST(payload);

  if (httpCode > 0) {
    String response = http.getString();
    Serial.print("HTTP Status: ");
    Serial.println(httpCode);
    Serial.print("ML Prediction Response: ");
    Serial.println(response);

    // Visual Alert indicator on ESP32
    if (response.indexOf("\"alert\":true") != -1 || response.indexOf("\"CRITICAL\"") != -1) {
      Serial.println("🚨 [LOCAL ALERT] Severe fault detected by ML model!");
      digitalWrite(STATUS_LED_PIN, HIGH);
    } else {
      digitalWrite(STATUS_LED_PIN, LOW);
    }
  } else {
    Serial.print("HTTP POST failed: ");
    Serial.println(http.errorToString(httpCode));
  }

  http.end();
}

// ================= SETUP =================
void setup() {
  Serial.begin(115200);
  delay(1000);

  Serial.println();
  Serial.println("==================================================");
  Serial.println("   SMARTMILL ESP32 PREDICTIVE MAINTENANCE NODE    ");
  Serial.println("==================================================");

  pinMode(VIBRATION_PIN, INPUT);
  pinMode(STATUS_LED_PIN, OUTPUT);
  digitalWrite(STATUS_LED_PIN, LOW);

  analogReadResolution(12);

  temperatureSensor.begin();
  Serial.print("DS18B20 sensors detected: ");
  Serial.println(temperatureSensor.getDeviceCount());

  connectWiFi();

  Serial.println("SmartMill initialization complete. Streaming to ML server...");
}

// ================= MAIN LOOP =================
void loop() {
  if (millis() - lastSendTime >= SEND_INTERVAL) {
    lastSendTime = millis();

    float temperature = readTemperature();
    int vibration = readVibration();
    int voltageADC = readVoltageADC();

    // 0-25V module calculation for local serial monitoring
    float approxVolts = (voltageADC / 4095.0) * 25.0;

    Serial.println();
    Serial.println("------ LIVE SENSOR DATA ------");
    Serial.print("Temperature: ");
    if (isnan(temperature)) {
      Serial.println("UNAVAILABLE");
    } else {
      Serial.print(temperature, 2);
      Serial.println(" °C");
    }

    Serial.print("Vibration (Digital pin 27): ");
    Serial.println(vibration == 1 ? "1 (TRIGGERED)" : "0 (STEADY)");

    Serial.print("Voltage ADC: ");
    Serial.print(voltageADC);
    Serial.print(" (~");
    Serial.print(approxVolts, 2);
    Serial.println(" V DC)");

    sendSensorData(temperature, vibration, voltageADC);
    Serial.println("------------------------------");
  }
}
