#include <accelerometers.h>
#include <SPI.h>
#include <Adafruit_Sensor.h>

#include <global_vars.h>

Adafruit_LIS3DH lis1 = Adafruit_LIS3DH(LIS3DH1_CS, &SPI, 2000000); // SPI speed 2MHz
Adafruit_LIS3DH lis2 = Adafruit_LIS3DH(LIS3DH2_CS, &hspi, 2000000); // SPI speed 2MHz
SPIClass hspi(HSPI); // Use VSPI for hardware SPI

AccelerometerData data1[NUM_SAMPLES];
AccelerometerData data2[NUM_SAMPLES];

void accel_setup (Adafruit_LIS3DH &lis, const uint8_t id, lis3dh_range_t range,
                 lis3dh_mode_t performance, lis3dh_dataRate_t dataRate) {
    if (! lis.begin()) {   // change this to 0x19 for alternative i2c address (default:0x18)
      Serial.print("Couldnt start sensor ");Serial.println(id);
      while (1) yield();
    }
    Serial.print("LIS3DH ");Serial.print(id);Serial.println(" found!");

    lis.setRange(range);   // 2, 4, 8 or 16 G!
    lis.setPerformanceMode(performance); // normal, low power, or high res
    lis.setDataRate(dataRate); // 1, 10, 25, 50, 100, 200, 400, 1600, 5000 Hz
    Serial.print("Range: "); Serial.println(lis.getRange());
    Serial.print("Performance Mode: "); Serial.println(lis.getPerformanceMode());
    Serial.print("Data Rate: "); Serial.println(lis.getDataRate());
}
  
AccelerometerData sample_accelerometer(Adafruit_LIS3DH &lis, unsigned long ts) {
    AccelerometerData tmp = {0, 0, 0, ts}; // initialize to 0
    //return raw value of acceleration in 8 bit value
    lis.read();
  
    #ifdef DEBUG
      //  Print raw accelerometer values
      // Serial.print("Raw X: "); Serial.print(lis.x);
      // Serial.print(" Y: "); Serial.print(lis.y);
      // Serial.print(" Z: "); Serial.println(lis.z);
    #endif
  
    // Assign the read values to the struct
    tmp.x = lis.x;
    tmp.y = lis.y;
    tmp.z = lis.z;
    tmp.timestamp = ts;
    return tmp;
}

void print_accelerometer_data(const AccelerometerData* data, uint16_t num_samples) {
    for (size_t i = 0; i < num_samples; i++) {
      Serial.print("Timestamp: "); Serial.print(data[i].timestamp); Serial.print(" us\t");
      Serial.print("X: "); Serial.print(data[i].x); Serial.print("\t");
      Serial.print("Y: "); Serial.print(data[i].y); Serial.print("\t");
      Serial.print("Z: "); Serial.print(data[i].z); Serial.println();
    }
    Serial.println("===================================");
}