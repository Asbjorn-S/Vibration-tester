#include <wifi_mqtt.h>
#include <ArduinoJson.h>
#include <global_vars.h>
#include <calibration.h>
#include <testing.h>

WiFiClient espClient;
PubSubClient client(espClient);

// #define DEBUG

#define JSON_BUFFER_SIZE 4096 // Adjust size based on your needs

void publishData(uint16_t nsample) {
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
        } else {
            Serial.println("JSON serialization failed.");
        }
        
        // Attempt to publish with retry
        int retries = 3;
        bool success = false;
        
        while (retries > 0 && !success) {
          success = client.publish("vibration/data", jsonBuffer);
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
        delay(100);
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

void callback(char* topic, byte* message, unsigned int length) {
    Serial.print("Message arrived on topic: ");
    Serial.print(topic);
    Serial.print(". Message: ");
    String messageTemp;
    
    for (int i = 0; i < length; i++) {
      Serial.print((char)message[i]);
      messageTemp += (char)message[i];
    }
    Serial.println();

    if (String(topic) == "vibration/calibration") {
        if (messageTemp == "start") {
            Serial.println("Starting calibration...");
            calibrationComplete = false; // Reset calibration flag
            test_frequency_v_amplitude(); // Call the calibration function
            publishCalibrationData(); // Publish calibration data to MQTT
            Serial.println("Calibration complete.");
        } else {
            Serial.println("Unknown command. Use 'start'");
        }
      }
    if (calibrationComplete) {
      if (String(topic) == "vibration/test") {
          Serial.println("Topic received: " + String(topic));
          uint16_t frequency = messageTemp.toInt();
          run_test(frequencyToAmplitude(frequency)); // Call the test function with the frequency value
      } else if (String(topic) == "vibration/frequency") {
          Serial.println("Topic received: " + String(topic));
          double targetFrequency = messageTemp.toDouble();
          Serial.print("Setting frequency to: ");
          Serial.print(targetFrequency);
          Serial.println(" Hz");
          uint16_t amplitude = frequencyToAmplitude(targetFrequency);
          ledcWrite(motorPWMChannel, amplitude);
      } else if (String(topic) == "vibration/amplitude") {
          Serial.println("Topic received: " + String(topic));
          uint16_t amplitude = messageTemp.toInt();
          ledcWrite(motorPWMChannel, amplitude);
      } else if (String(topic) == "vibration/motor") { 
          Serial.println("Topic received: " + String(topic));
          if (messageTemp == "on") {
              digitalWrite(MOTOR_IN1, HIGH); // Turn on motor
              Serial.println("Motor turned ON");
          } else if (messageTemp == "off") {
              digitalWrite(MOTOR_IN1, LOW); // Turn off motor
              Serial.println("Motor turned OFF");
          } else {
              Serial.println("Invalid command. Use 'on' or 'off'.");
          }
      } else {
        Serial.println("Unknown topic. No action taken.");
      }
    } else {
        Serial.println("Calibration not complete. No action taken.");
        client.publish("vibration/calibration/status", "incomplete");
    }
  }

void setup_wifi() {
    delay(10);
    Serial.println();
    Serial.print("Connecting to ");
    Serial.println(SSID);
  
    WiFi.begin(SSID, PASSWORD);
  
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
        client.subscribe("vibration/test");
        client.subscribe("vibration/calibration");
        client.subscribe("vibration/frequency");
        client.subscribe("vibration/amplitude");
        client.subscribe("vibration/motor");
      } else {
        Serial.print("failed, rc=");
        Serial.print(client.state());
        Serial.println(" try again in 5 seconds");
        delay(5000);
      }
    }
  }