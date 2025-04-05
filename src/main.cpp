#include <ArduinoJson.h>
#include <global_vars.h>
#include <wifi_mqtt.h>
#include <accelerometers.h>
#include <calibration.h>


#define DEBUG // Uncomment to enable debug prints

// global variables and constants for the project
uint16_t amplitude = 1023; // default amplitude for motor PWM
bool calibrationComplete = false; // Flag to indicate if calibration is complete

#define JSON_BUFFER_SIZE 4096 // Adjust size based on your needs

bool running_test = false;

// Variables for motor control and sampling
unsigned long motor_start_time = 0;
unsigned long previousMicros = 0;

uint16_t nsample = 0;

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
          // Serial.print("Free heap: ");
          // Serial.println(ESP.getFreeHeap());
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
            success = client.publish(vibration_data_topic, jsonBuffer);
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
          // Serial.print("Serialized JSON size: ");
          // Serial.println(jsonSize);
          // Serial.print("Serialized JSON: ");
          // Serial.println(jsonBuffer);
          // Serial.print("Free heap: ");
          // Serial.println(ESP.getFreeHeap());
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
  client.setServer(MQTT_SERVER, 1883);
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
    String input = Serial.readStringUntil('\n');
    if (input.startsWith("cal")) {
      Serial.println("Starting calibration...");
      calibrationComplete = false; // Reset calibration flag
      test_frequency_v_amplitude(); // Call the calibration function
      Serial.println("Calibration complete.");
    } else if (calibrationComplete) {
      input.trim();
      if (input.startsWith("o")) {
        if (input == "on") {
          digitalWrite(MOTOR_IN1, HIGH); // Turn on motor
          Serial.println("Motor turned ON");
        } else if (input == "off") {
          digitalWrite(MOTOR_IN1, LOW); // Turn off motor
          Serial.println("Motor turned OFF");
        } else {
          Serial.println("Invalid command. Use 'on' or 'off'.");
        }
      } else if (input.startsWith("a:")){  // Treat as direct amplitude input
        amplitude = input.substring(2).toInt();
        if (amplitude > 1023) amplitude = 1023;
        if (amplitude < 0) amplitude = 0;
        ledcWrite(motorPWMChannel, amplitude);
        Serial.print("Amplitude: "); Serial.println(amplitude);
      } else {
        double targetFrequency = input.toDouble();
        Serial.print("Setting frequency to: ");
        Serial.print(targetFrequency);
        Serial.println(" Hz");
        amplitude = frequencyToAmplitude(targetFrequency);
        ledcWrite(motorPWMChannel, amplitude);
      }
    } else {
    Serial.println("Calibration not complete. Press button or type \"cal\" to start calibration.");
  }
}
  // before starting the test sequence, calibrate the motor
  if (digitalRead(BUTTONPIN)) {
    test_frequency_v_amplitude();
    // Print frequency data for debugging
    #ifdef DEBUG
      Serial.println("Frequency data:");
      for (uint16_t i = 0; i <= (maxAmplitude-minAmplitude)/CAL_STEP; i++) {
        Serial.print(" -> Frequency: ");
        Serial.print(freqData[i].frequency);
        Serial.print(" Hz, Amplitude: ");
        Serial.println(freqData[i].amplitude);
      }
    #endif
  }
  
  // Run test sequence when button is pressed
  // run_test_sequence();
}