#pragma once

#include <global_vars.h>
#include <arduinoFFT.h>

// FFT settings
#define FFT_SAMPLES 1024 // Number of samples for FFT
#define CAL_STEP 100 // Step size for calibration

extern double vReal[FFT_SAMPLES]; // Real part of the FFT input
extern double vImag[FFT_SAMPLES]; // Imaginary part of the FFT input
extern ArduinoFFT<double> FFT; // Initialize FFT object

struct frequencyData {
  float frequency;
  uint16_t amplitude;
};
extern frequencyData freqData[(maxAmplitude-minAmplitude)/CAL_STEP+2];

void test_frequency_v_amplitude();

void publishCalibrationData();

uint16_t frequencyToAmplitude(float targetFrequency);
