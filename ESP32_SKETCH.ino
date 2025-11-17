/*
 * ESP32 Smart Door Lock - BLE Server with Reed Sensor
 * 
 * This sketch:
 * 1. Creates a BLE Server (advertises as "ESP32_DoorLock")
 * 2. Listens for OPEN/CLOSE commands from Flutter app
 * 3. Controls relay to lock/unlock door
 * 4. Monitors reed sensor for forced entry
 * 5. Sends forced entry events to Flutter app
 * 
 * Pins:
 * GPIO 5  - Relay Control (HIGH = OPEN, LOW = CLOSED)
 * GPIO 4  - Reed Sensor (HIGH = open, LOW = closed)
 * GPIO 2  - LED Status indicator
 */

#include <BLEDevice.h>
#include <BLEServer.h>
#include <BLEUtils.h>
#include <BLE2902.h>
#include <WiFi.h>
#include <HTTPClient.h>

// ============================================================================
// PIN DEFINITIONS
// ============================================================================
#define RELAY_PIN 5           // Controls door lock relay
#define REED_SENSOR_PIN 4     // Magnetic reed sensor
#define LED_PIN 2             // Status LED
#define BUZZER_PIN 14         // Buzzer for alert

// ============================================================================
// BLE CONFIGURATION
// ============================================================================
#define SERVICE_UUID        "12345678-1234-5678-1234-56789abcdef0"
#define CHARACTERISTIC_UUID "87654321-4321-8765-4321-0fedcba98765"

BLEServer *pServer = NULL;
BLECharacteristic *pCharacteristic = NULL;
bool deviceConnected = false;
bool oldDeviceConnected = false;

// ============================================================================
// COMMAND DEFINITIONS
// ============================================================================
const char* CMD_OPEN = "OPEN";
const char* CMD_CLOSE = "CLOSE";

// Backend configuration
const char* BACKEND_IP = "192.168.1.100";  // Change to your PC IP
const int BACKEND_PORT = 8000;

// ============================================================================
// CALLBACK CLASS FOR BLE COMMANDS
// ============================================================================

class MyServerCallbacks: public BLEServerCallbacks {
    void onConnect(BLEServer* pServer) {
      deviceConnected = true;
      digitalWrite(LED_PIN, HIGH);  // LED on when connected
      Serial.println("Client connected!");
    };

    void onDisconnect(BLEServer* pServer) {
      deviceConnected = false;
      digitalWrite(LED_PIN, LOW);   // LED off when disconnected
      Serial.println("Client disconnected!");
    }
};

class MyCharacteristicCallbacks: public BLECharacteristicCallbacks {
    void onWrite(BLECharacteristic *pCharacteristic) {
      std::string value = pCharacteristic->getValue();
      
      if (value.length() > 0) {
        Serial.print("Received command: ");
        for (int i = 0; i < value.length(); i++) {
          Serial.print(value[i]);
        }
        Serial.println();

        // Handle OPEN command
        if (value == CMD_OPEN) {
          Serial.println("🔓 Opening door...");
          digitalWrite(RELAY_PIN, HIGH);  // Activate relay
          digitalWrite(LED_PIN, HIGH);    // Turn LED on
          delay(3000);                    // Keep relay active for 3 seconds
          digitalWrite(RELAY_PIN, LOW);   // Release relay
          pCharacteristic->setValue("DOOR_OPENED");
        }
        // Handle CLOSE command
        else if (value == CMD_CLOSE) {
          Serial.println("🔒 Closing door...");
          digitalWrite(RELAY_PIN, LOW);   // Deactivate relay
          digitalWrite(LED_PIN, LOW);     // Turn LED off
          pCharacteristic->setValue("DOOR_CLOSED");
        }
        else {
          Serial.println("⚠️ Unknown command!");
          pCharacteristic->setValue("ERROR");
        }
      }
    }
};

// ============================================================================
// REED SENSOR MONITORING
// ============================================================================

void checkReedSensor() {
  static int lastReedState = LOW;
  int currentReedState = digitalRead(REED_SENSOR_PIN);

  // Detect state change (door opened)
  if (currentReedState != lastReedState) {
    delay(50);  // Debounce
    currentReedState = digitalRead(REED_SENSOR_PIN);

    if (currentReedState == HIGH) {
      // Reed sensor opened (door opened)
      Serial.println("⚠️ FORCED ENTRY DETECTED!");
      triggerForcedEntryAlert();
    }
    lastReedState = currentReedState;
  }
}

void triggerForcedEntryAlert() {
  // Buzz buzzer
  for (int i = 0; i < 3; i++) {
    digitalWrite(BUZZER_PIN, HIGH);
    delay(500);
    digitalWrite(BUZZER_PIN, LOW);
    delay(500);
  }

  // Blink LED rapidly
  for (int i = 0; i < 10; i++) {
    digitalWrite(LED_PIN, !digitalRead(LED_PIN));
    delay(100);
  }

  // Send alert to backend
  logForcedEntryToBackend();
}

void logForcedEntryToBackend() {
  if (WiFi.status() == WL_CONNECTED) {
    HTTPClient http;
    String url = "http://" + String(BACKEND_IP) + ":" + String(BACKEND_PORT) + "/log_forced_entry/";
    
    http.begin(url);
    http.addHeader("Content-Type", "application/json");
    
    String payload = "{\"timestamp\": \"" + String(millis()) + "\"}";
    int httpResponseCode = http.POST(payload);
    
    if (httpResponseCode > 0) {
      Serial.println("Forced entry logged to backend. Response code: " + String(httpResponseCode));
    } else {
      Serial.println("Failed to log forced entry. Error: " + String(http.errorToString(httpResponseCode)));
    }
    http.end();
  }
}

// ============================================================================
// SETUP
// ============================================================================

void setup() {
  Serial.begin(115200);
  
  // Initialize pins
  pinMode(RELAY_PIN, OUTPUT);
  pinMode(REED_SENSOR_PIN, INPUT_PULLUP);
  pinMode(LED_PIN, OUTPUT);
  pinMode(BUZZER_PIN, OUTPUT);

  // Default states
  digitalWrite(RELAY_PIN, LOW);
  digitalWrite(LED_PIN, LOW);
  digitalWrite(BUZZER_PIN, LOW);

  Serial.println("\n\n=== Smart Door Lock Starting ===\n");

  // Initialize BLE
  BLEDevice::init("ESP32_DoorLock");
  pServer = BLEDevice::createServer();
  pServer->setCallbacks(new MyServerCallbacks());

  // Create BLE Service
  BLEService *pService = pServer->createService(SERVICE_UUID);

  // Create BLE Characteristic
  pCharacteristic = pService->createCharacteristic(
    CHARACTERISTIC_UUID,
    BLECharacteristic::PROPERTY_READ |
    BLECharacteristic::PROPERTY_WRITE |
    BLECharacteristic::PROPERTY_NOTIFY |
    BLECharacteristic::PROPERTY_INDICATE
  );

  // Add callbacks
  pCharacteristic->setCallbacks(new MyCharacteristicCallbacks());

  // Create a BLE Descriptor
  pCharacteristic->addDescriptor(new BLE2902());

  // Set initial value
  pCharacteristic->setValue("DOOR_CLOSED");

  // Start service
  pService->start();

  // Start advertising
  BLEAdvertising *pAdvertising = BLEDevice::getAdvertising();
  pAdvertising->addServiceUUID(SERVICE_UUID);
  pAdvertising->setScanResponse(false);
  pAdvertising->setMinPreferred(0x0);  // set value to 0x00 to not advertise this parameter
  BLEDevice::startAdvertising();

  Serial.println("BLE Server started. Device name: ESP32_DoorLock");
  Serial.println("Service UUID: " + String(SERVICE_UUID));
  Serial.println("Characteristic UUID: " + String(CHARACTERISTIC_UUID));
  Serial.println("\nWaiting for connections...");
}

// ============================================================================
// MAIN LOOP
// ============================================================================

void loop() {
  // Check for new connections
  if (!deviceConnected && oldDeviceConnected) {
    delay(500);
    pServer->startAdvertising();
    Serial.println("Start advertising");
    oldDeviceConnected = deviceConnected;
  }

  if (deviceConnected && !oldDeviceConnected) {
    oldDeviceConnected = deviceConnected;
  }

  // Monitor reed sensor for forced entry
  checkReedSensor();

  // Small delay
  delay(100);
}

/*
 * ARDUINO IDE INSTALLATION:
 * 
 * 1. Install "ESP32" board in Boards Manager
 * 2. Select Board: "ESP32 Dev Module"
 * 3. Select COM Port (where ESP32 is connected)
 * 4. Paste this code into Arduino IDE
 * 5. Update BACKEND_IP to your PC's IP address
 * 6. Click Upload
 * 
 * DEBUGGING:
 * - Open Serial Monitor (Baud: 115200)
 * - You should see "BLE Server started"
 * - When Flutter app connects, you'll see "Client connected"
 * - When you click "Lock"/"Unlock" in app, relay activates
 * 
 * TROUBLESHOOTING:
 * - No "BLE Server started": Check ESP32 driver
 * - Device not found in Flutter app: Restart ESP32, check name
 * - Relay not activating: Check GPIO 5 pin connection and relay wiring
 */
