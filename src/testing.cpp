#include <testing.h>
#include <Arduino.h>
#include <global_vars.h>
#include <accelerometers.h>
#include <ArduinoJson.h>
#include <wifi_mqtt.h>
#include <calibration.h>

void run_test(double frequency) {
    Serial.println("Starting test sequence with frequency " + String(frequency));
    // Start motor
    ledcWrite(motorPWMChannel, frequencyToAmplitude(frequency)); // Set motor amplitude
    digitalWrite(MOTOR_IN1, HIGH);

    uint16_t nsample = 0;
    unsigned long motor_start_time = millis();
    unsigned long previousMicros = micros(); // reset previousMicros to current time
    while (millis() - motor_start_time < 500) {} // wait for 500ms before starting sampling
    while (true) { // Run for the test duration
      unsigned long timestamp = micros();
      if (timestamp - previousMicros >= SAMPLE_TIME) {
        previousMicros += SAMPLE_TIME; // increment by sample time to avoid drift
        if (nsample < NUM_SAMPLES) {
            // Sample accelerometer data
            data1[nsample] = sample_accelerometer(lis1);
            data2[nsample] = sample_accelerometer(lis2);
            nsample++;
        } else {
            Serial.println("Max samples reached, stopping test");
            digitalWrite(MOTOR_IN1, LOW);
            publishData(nsample, frequency);
            break;
        }
      }
    }
    // Stop motor after the test
}
  