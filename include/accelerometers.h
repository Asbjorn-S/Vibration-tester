#pragma once

#include <Adafruit_LIS3DH.h>
#include <global_vars.h>

// Define the pins for the LIS3DH accelerometers
#define LIS3DH1_CS 5 // Chip select pin for first LIS3DH
#define LIS3DH2_CS 27 // Chip select pin for second LIS3DH
#define HSPI_SCK 14   // HSPI Clock
#define HSPI_MISO 12  // HSPI MISO
#define HSPI_MOSI 13  // HSPI MOSI

extern SPIClass hspi; // Use HSPI for second accelerometer
extern Adafruit_LIS3DH lis1; // VSPI
extern Adafruit_LIS3DH lis2; // HSPI speed 2MHz


void accel_setup(Adafruit_LIS3DH &lis, const uint8_t id, lis3dh_range_t range = LIS3DH_RANGE_8_G, lis3dh_mode_t performance = LIS3DH_MODE_LOW_POWER ,lis3dh_dataRate_t dataRate = LIS3DH_DATARATE_LOWPOWER_1K6HZ);

AccelerometerData sample_accelerometer(Adafruit_LIS3DH &lis);

void print_accelerometer_data(const AccelerometerData *data, uint16_t num_samples);
