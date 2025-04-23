#pragma once

#include <WiFi.h>
#include <PubSubClient.h>

#define MQTT_MAX_PACKET_SIZE 4096  // Increase this value based on your payload size

// WiFi and MQTT configuration
#define SSID "iOT Deco"         // Replace with your WiFi SSID
#define PASSWORD "bre-rule-247"   // Replace with your WiFi password
#define MQTT_SERVER "192.168.68.126" // Replace with your MQTT broker address

#define vibration_data_topic "vibration/data" // MQTT topic to publish data

extern WiFiClient espClient;
extern PubSubClient client;

void publishData(uint16_t nsample, double frequency);

void callback(char *topic, byte *message, unsigned int length);

void setup_wifi();

void reconnect();
