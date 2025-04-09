#pragma once

#include <Arduino.h>

// Global variables and constants for the project
#define BUTTONPIN 26
#define MOTOR_PWM 33
#define MOTOR_IN1 32

#define motorPWMFreq 5000
#define motorPWMChannel 0

// Motor PWM settings
const uint16_t minAmplitude = 300; // Minimum PWM amplitude
const uint16_t maxAmplitude = 1023; // Maximum PWM amplitude

// Sampling settings
#define SAMPLE_TIME 625 // 625 microseconds = 1.6kHz [us]
#define SAMPLE_FREQ 1000000/SAMPLE_TIME // Sampling frequency [Hz]
#define TEST_TIME 500000 //  Time for test sequence [us]
#define NUM_SAMPLES TEST_TIME/SAMPLE_TIME // number of samples to take

extern bool calibrationComplete; // Flag to indicate if calibration is complete

// Accelerometer data structure
struct AccelerometerData {
    float x;
    float y;
    // float z;
    unsigned long timestamp;
};
extern AccelerometerData data1[NUM_SAMPLES];
extern AccelerometerData data2[NUM_SAMPLES];
