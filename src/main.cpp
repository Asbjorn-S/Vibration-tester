#include <Wire.h>
#include <SPI.h>
#include <Adafruit_LIS3DH.h>
#include <Adafruit_Sensor.h>
#include <WiFi.h>
#include <PubSubClient.h>
#include <ArduinoJson.h>
#include <algorithm>
#include <arduinoFFT.h>

#define DEBUG // Uncomment to enable debug prints

// WiFi and MQTT configuration
const char* ssid = "iOT Deco";           // Replace with your WiFi SSID
const char* password = "bre-rule-247";   // Replace with your WiFi password
const char* mqtt_server = "192.168.68.103"; // Replace with your MQTT broker address

WiFiClient espClient;
PubSubClient client(espClient);

#define MQTT_MAX_PACKET_SIZE 4096  // Increase this value based on your payload size
#define JSON_BUFFER_SIZE 4096 // Adjust size based on your needs

const char* mqtt_topic = "vibration/data"; // MQTT topic to publish data

// Define the pins for the LIS3DH accelerometers
#define LIS3DH1_CS 5 // Chip select pin for first LIS3DH
#define LIS3DH2_CS 27 // Chip select pin for second LIS3DH
#define HSPI_SCK 14   // HSPI Clock
#define HSPI_MISO 12  // HSPI MISO
#define HSPI_MOSI 13  // HSPI MOSI
SPIClass hspi(HSPI); // Use VSPI for hardware SPI
Adafruit_LIS3DH lis1 = Adafruit_LIS3DH(LIS3DH1_CS, &SPI, 2000000); // SPI speed 2MHz
Adafruit_LIS3DH lis2 = Adafruit_LIS3DH(LIS3DH2_CS, &hspi, 2000000); // SPI speed 2MHz

// Variables for motor control and sampling
unsigned long motor_start_time = 0;
unsigned long previousMicros = 0;

bool running_test = false;

// Define the pins for the motor and button
#define BUTTONPIN 26
#define MOTOR_PWM 33
#define MOTOR_IN1 32
#define motorPWMFreq 5000
#define motorPWMChannel 0

// Motor PWM settings
const uint16_t minAmplitude = 250; // Minimum PWM amplitude
const uint16_t maxAmplitude = 1023; // Maximum PWM amplitude
uint16_t amplitude = 1023; // default amplitude for motor PWM

// Sampling settings
#define SAMPLE_TIME 625 // 625 microseconds = 1.6kHz [us]
#define SAMPLE_FREQ 1000000/SAMPLE_TIME // Sampling frequency [Hz]
#define TEST_TIME 500000 //  Time for test sequence [us]
#define NUM_SAMPLES TEST_TIME/SAMPLE_TIME // number of samples to take

uint16_t nsample = 0;

// FFT settings
#define FFT_SAMPLES 1024 // Number of samples for FFT
double vReal[FFT_SAMPLES]; // Real part of the FFT input
double vImag[FFT_SAMPLES]; // Imaginary part of the FFT input
ArduinoFFT<double> FFT = ArduinoFFT<double>(vReal, vImag, FFT_SAMPLES, SAMPLE_FREQ); // Initialize FFT object

struct AccelerometerData {
  int x;
  int y;
  int z;
  unsigned long timestamp;
};
AccelerometerData data1[NUM_SAMPLES];
AccelerometerData data2[NUM_SAMPLES];

void setup_wifi() {
  delay(10);
  Serial.println();
  Serial.print("Connecting to ");
  Serial.println(ssid);

  WiFi.begin(ssid, password);

  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }

  Serial.println();
  Serial.println("WiFi connected");
  Serial.println("IP address: ");
  Serial.println(WiFi.localIP());
}

void reconnect() {
  // Loop until we're reconnected
  while (!client.connected()) {
    Serial.print("Attempting MQTT connection...");
    // Attempt to connect
    if (client.connect("VibrationClient")) {
      Serial.println("connected");
    } else {
      Serial.print("failed, rc=");
      Serial.print(client.state());
      Serial.println(" try again in 5 seconds");
      delay(5000);
    }
  }
}

void accel_setup (Adafruit_LIS3DH &lis, const uint8_t id, lis3dh_range_t range = LIS3DH_RANGE_8_G, lis3dh_mode_t performance = LIS3DH_MODE_LOW_POWER ,lis3dh_dataRate_t dataRate = LIS3DH_DATARATE_LOWPOWER_1K6HZ) {
  if (! lis.begin()) {   // change this to 0x19 for alternative i2c address (default:0x18)
    Serial.print("Couldnt start sensor ");Serial.println(id);
    while (1) yield();
  }
  Serial.print("LIS3DH ");Serial.print(id);Serial.println(" found!");

  lis.setRange(range);   // 2, 4, 8 or 16 G!
  lis.setPerformanceMode(performance); // normal, low power, or high res
  lis.setDataRate(dataRate); // 1, 10, 25, 50, 100, 200, 400, 1600, 5000 Hz
  Serial.print("Range: "); Serial.println(lis.getRange());
  Serial.print("Performance Mode: "); Serial.println(lis.getPerformanceMode());
  Serial.print("Data Rate: "); Serial.println(lis.getDataRate());
}

AccelerometerData sample_accelerometer(Adafruit_LIS3DH &lis, unsigned long ts) {
  AccelerometerData tmp = {0, 0, 0, ts}; // initialize to 0
  //return raw value of acceleration in 8 bit value
  lis.read();

  #ifdef DEBUG
    //  Print raw accelerometer values
    // Serial.print("Raw X: "); Serial.print(lis.x);
    // Serial.print(" Y: "); Serial.print(lis.y);
    // Serial.print(" Z: "); Serial.println(lis.z);
  #endif

  // Assign the read values to the struct
  tmp.x = lis.x;
  tmp.y = lis.y;
  tmp.z = lis.z;
  tmp.timestamp = ts;
  return tmp;
}

void print_accelerometer_data(const AccelerometerData* data, uint16_t num_samples) {
  for (size_t i = 0; i < num_samples; i++) {
    Serial.print("Timestamp: "); Serial.print(data[i].timestamp); Serial.print(" us\t");
    Serial.print("X: "); Serial.print(data[i].x); Serial.print("\t");
    Serial.print("Y: "); Serial.print(data[i].y); Serial.print("\t");
    Serial.print("Z: "); Serial.print(data[i].z); Serial.println();
  }
  Serial.println("===================================");
}

void calibrate_vibration_motor() {
  // Placeholder for motor calibration logic
  // This function can be used to calibrate the motor based on the accelerometer data
  Serial.println("Calibrating vibration motor...");
  // Add calibration logic here
  const uint16_t step = 50; // Step size for amplitude
  // const uint16_t calibrationTime = 1000; // Time to measure frequency at each amplitude [ms]

  ledcWrite(motorPWMChannel, minAmplitude); // Set initial amplitude
  digitalWrite(MOTOR_IN1, HIGH); // Start the motor



  for (uint16_t amplitude = minAmplitude; amplitude < maxAmplitude + step; amplitude += step) {
    if (amplitude > maxAmplitude) amplitude = maxAmplitude; // Ensure amplitude doesn't exceed maxAmplitude
    // Set motor amplitude
    ledcWrite(motorPWMChannel, amplitude);
    // Wait for the motor to stabilize
    delay(100); // Wait for 100 ms to stabilize

    // Measure frequency
    unsigned long startTime = millis();
    uint16_t sampleIndex = 0;

    while (sampleIndex < FFT_SAMPLES) {
      unsigned long timestamp = micros();
      AccelerometerData sample = sample_accelerometer(lis1, timestamp);

      // Store the accelerometer's X-axis data in the FFT input array
      vReal[sampleIndex] = sample.x; // Magnitude of acceleration vector
      vImag[sampleIndex] = 0; // Imaginary part is zero for real input
      sampleIndex++;

      // Wait for the next sample (based on the sampling frequency)
      delayMicroseconds(SAMPLE_TIME);
    }
    
    #ifdef DEBUG
    // print vReal for debugging
      // Serial.print("vReal: ");
      // for (uint16_t i = 0; i < FFT_SAMPLES; i++) {
      //   Serial.print(vReal[i]);
      //   Serial.print(" ");
      // }
      // Serial.println();
    #endif

    // Perform FFT
    FFT.windowing(FFT_WIN_TYP_HAMMING, FFT_FORWARD); // Apply a Hamming window
    FFT.compute(FFT_FORWARD); // Compute the FFT
    FFT.complexToMagnitude(); // Compute magnitudes

    double peakFrequency = 0;
    double maxMagnitude = 0;
    for (uint16_t i = 1; i < (FFT_SAMPLES / 2); i++) { // Ignore DC component at index 0
      if (vReal[i] > maxMagnitude) {
        maxMagnitude = vReal[i];
        peakFrequency = i * (SAMPLE_FREQ / FFT_SAMPLES);
      }
    }

    // Print the result for debugging
    #ifdef DEBUG
      Serial.print("Amplitude: ");
      Serial.print(amplitude);
      Serial.print(" -> Frequency: ");
      Serial.print(peakFrequency);
      Serial.println(" Hz");
    #endif
  }
  // Turn off the motor after calibration
  digitalWrite(MOTOR_IN1, LOW);
  ledcWrite(motorPWMChannel, minAmplitude); // Set amplitude to minimum 
  Serial.println("Calibration complete.");
}

void run_test_sequence() {
  if (digitalRead(BUTTONPIN) || running_test) {
    if (!running_test) {
      Serial.println("Button pressed, starting test sequence");
      running_test = true;
      // Start motor
      digitalWrite(MOTOR_IN1, HIGH);
      motor_start_time = millis();
      #ifdef DEBUG
        Serial.print("Motor start time: ");
        Serial.println(motor_start_time);
      #endif
    }

    previousMicros = micros(); // reset previousMicros to current time
    if (millis() - motor_start_time > 500) { // wait for 500ms before starting sampling
      while (true) { // Run for the test duration
        unsigned long timestamp = micros();
      
        if (timestamp - previousMicros >= SAMPLE_TIME) {
          previousMicros += SAMPLE_TIME; // increment by sample time to avoid drift
        
          if (nsample < NUM_SAMPLES) {
            // Sample accelerometer data
            data1[nsample] = sample_accelerometer(lis1, timestamp);
            data2[nsample] = sample_accelerometer(lis2, timestamp);
            nsample++;
          } else {
            Serial.println("Max samples reached, stopping test sequence");
            break;
          }
        }
      }
    
      // Stop motor after the test
      digitalWrite(MOTOR_IN1, LOW);
      running_test = false;

      // Serialize and publish the dataset in chunks
      const size_t chunkSize = 30; // Number of samples per chunk
      for (uint16_t start = 0; start < nsample; start += chunkSize) {
        
        #ifdef DEBUG
          // Print the current chunk size and free heap memory
          Serial.print("Free heap: ");
          Serial.println(ESP.getFreeHeap());
        #endif

        uint16_t end = std::min<uint16_t>(start + chunkSize, nsample);

        // Use DynamicJsonDocument for flexibility
        DynamicJsonDocument doc(JSON_BUFFER_SIZE); // Adjust size based on chunk size
        JsonArray accel1Array = doc.createNestedArray("accelerometer1");
        JsonArray accel2Array = doc.createNestedArray("accelerometer2");

        for (uint16_t i = start; i < end; i++) {
          JsonObject sample1Obj = accel1Array.createNestedObject();
          sample1Obj["x"] = data1[i].x;
          sample1Obj["y"] = data1[i].y;
          sample1Obj["z"] = data1[i].z;
          sample1Obj["timestamp"] = data1[i].timestamp;

          JsonObject sample2Obj = accel2Array.createNestedObject();
          sample2Obj["x"] = data2[i].x;
          sample2Obj["y"] = data2[i].y;
          sample2Obj["z"] = data2[i].z;
          sample2Obj["timestamp"] = data2[i].timestamp;
        }

        // Add chunk information
        doc["chunk"] = start / chunkSize + 1;
        doc["totalChunks"] = (nsample + chunkSize - 1) / chunkSize;

        // Serialize the JSON document to a string
        char jsonBuffer[JSON_BUFFER_SIZE]; // Adjust size based on chunk size
        size_t jsonSize = serializeJson(doc, jsonBuffer, sizeof(jsonBuffer));

        if (jsonSize > 0) {
          // Check if the client is connected
          if (!client.connected()) {
            Serial.println("MQTT client disconnected. Reconnecting...");
            reconnect();
          }
          
          // Attempt to publish with retry
          int retries = 3;
          bool success = false;
          
          while (retries > 0 && !success) {
            success = client.publish(mqtt_topic, jsonBuffer);
            if (success) {
              #ifdef DEBUG
                Serial.print("Chunk ");
                Serial.print(doc["chunk"].as<int>());
                Serial.print("/");
                Serial.print(doc["totalChunks"].as<int>());
                Serial.println(" published successfully.");
                #endif
            } else {
              Serial.print("Failed to publish chunk. Retrying... (");
              Serial.print(retries);
              Serial.println(" attempts left)");
              delay(100); // Wait before retrying
              retries--;
            }
          }
          
          if (!success) {
            Serial.println("Failed to publish after multiple attempts.");
          }
          
          // Small delay between publishing chunks to avoid flooding the broker
          delay(50);
        } else {
          Serial.println("Failed to serialize JSON chunk.");
        }
        #ifdef DEBUG
          //  Print the serialized JSON, its size and free heap memory
          Serial.print("Serialized JSON size: ");
          Serial.println(jsonSize);
          Serial.print("Serialized JSON: ");
          Serial.println(jsonBuffer);
          Serial.print("Free heap: ");
          Serial.println(ESP.getFreeHeap());
        #endif
      }
    }
    nsample = 0; // Reset sample counter for the next test
  } 
}

void setup(void) {
  Serial.begin(115200);
  //while (!Serial) delay(10);     // will pause Zero, Leonardo, etc until serial console opens

  // Connect to WiFi
  setup_wifi();

  // Set up MQTT
  client.setBufferSize(MQTT_MAX_PACKET_SIZE);
  client.setServer(mqtt_server, 1883);
  client.setKeepAlive(60); // Set keep-alive interval to 60 seconds

  // Test simple MQTT publish after connection
  if (client.connect("VibrationClientTest")) {
    Serial.println("Testing MQTT connection with small message...");
    bool testSuccess = client.publish("vibration/test", "Hello World");
    if (testSuccess) {
      Serial.println("Test message published successfully!");
    } else {
      Serial.println("Failed to publish test message. Check broker configuration.");
    }
  }

  accel_setup(lis1, 1);
  accel_setup(lis2, 2);

  pinMode(BUTTONPIN, INPUT_PULLDOWN); // button to trigger motor
  pinMode(MOTOR_IN1, OUTPUT); // enable pin for motor

  // Set up PWM for motor
  ledcSetup(motorPWMChannel, motorPWMFreq, 10);
  ledcAttachPin(MOTOR_PWM, motorPWMChannel);
  ledcWrite(motorPWMChannel, amplitude);

  Serial.println("Setup complete");
}

void loop() {
  if (!client.connected()) {
    reconnect();
  }
  client.loop();

  // listen for input from serial to set target frequency for vibration motor
  if (Serial.available() > 0) {
    amplitude = Serial.parseInt();
    if (amplitude > 1023) amplitude = 1023;
    if (amplitude < 0) amplitude = 0;
    ledcWrite(motorPWMChannel, amplitude);
    Serial.print("Amplitude: "); Serial.println(amplitude);
  }
  // before starting the test sequence, calibrate the motor
  // if (digitalRead(BUTTONPIN)) {
  //   calibrate_vibration_motor();
  // }
  // Run test sequence when button is pressed
  run_test_sequence();
}