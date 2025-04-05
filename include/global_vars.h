#pragma once
// Sampling settings
#define SAMPLE_TIME 625 // 625 microseconds = 1.6kHz [us]
#define SAMPLE_FREQ 1000000/SAMPLE_TIME // Sampling frequency [Hz]
#define TEST_TIME 500000 //  Time for test sequence [us]
#define NUM_SAMPLES TEST_TIME/SAMPLE_TIME // number of samples to take
