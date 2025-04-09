#include <ArduinoJson.h>
#include <global_vars.h>
#include <wifi_mqtt.h>
#include <accelerometers.h>
#include <calibration.h>

// #define DEBUG // Uncomment to enable debug prints

// global variables and constants for the project
bool calibrationComplete = false; // Flag to indicate if calibration is complete
bool firstRun = true; // Flag to indicate if this is the first run of the program

void setup(void) {
  Serial.begin(115200);

  // Connect to WiFi
  setup_wifi();

  // Set up MQTT
  client.setBufferSize(MQTT_MAX_PACKET_SIZE);
  client.setServer(MQTT_SERVER, 1883);
  client.setCallback(callback);
  client.setKeepAlive(60); // Set keep-alive interval to 60 seconds

  accel_setup(lis1, 1, LIS3DH_RANGE_16_G);
  accel_setup(lis2, 2);

  pinMode(BUTTONPIN, INPUT_PULLDOWN); // button to trigger motor
  pinMode(MOTOR_IN1, OUTPUT); // enable pin for motor

  // Set up PWM for motor
  ledcSetup(motorPWMChannel, motorPWMFreq, 10);
  ledcAttachPin(MOTOR_PWM, motorPWMChannel);

  Serial.println("Setup complete");
}

void loop() {
  if (!client.connected()) {
    reconnect();
  }
  client.loop();

  if (firstRun) {
    firstRun = false; // Set the flag to false after the first run
    client.publish("vibration/calibration/status", "incomplete");
  }
}