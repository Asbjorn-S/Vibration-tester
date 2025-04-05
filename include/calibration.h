#pragma once

#include <global_vars.h>
#include <arduinoFFT.h>

#define motorPWMFreq 5000
#define motorPWMChannel 0

// FFT settings
#define FFT_SAMPLES 1024 // Number of samples for FFT
#define CAL_STEP 100 // Step size for calibration

extern double vReal[FFT_SAMPLES]; // Real part of the FFT input
extern double vImag[FFT_SAMPLES]; // Imaginary part of the FFT input
extern ArduinoFFT<double> FFT; // Initialize FFT object

struct frequencyData {
  double frequency;
  uint16_t amplitude;
};
extern frequencyData freqData[(maxAmplitude-minAmplitude)/CAL_STEP+2];

void test_frequency_v_amplitude();

uint16_t frequencyToAmplitude(double targetFrequency);
