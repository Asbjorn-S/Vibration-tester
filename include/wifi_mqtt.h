#pragma once

#include <WiFi.h>
#include <PubSubClient.h>

#define MQTT_MAX_PACKET_SIZE 4096  // Increase this value based on your payload size

// WiFi and MQTT configuration
#define SSID "Sorensen"         // Replace with your WiFi SSID
#define PASSWORD "Kartoffel"   // Replace with your WiFi password
#define MQTT_SERVER "192.168.0.154" // Replace with your MQTT broker address

#define vibration_data_topic "vibration/data" // MQTT topic to publish data

extern WiFiClient espClient;
extern PubSubClient client;

void setup_wifi();

void reconnect();
