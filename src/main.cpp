#include <Wire.h>
#include <SPI.h>
#include <Adafruit_LIS3DH.h>
#include <Adafruit_Sensor.h>

// Used for software SPI
// #define LIS3DH_CLK 13
// #define LIS3DH_MISO 12
// #define LIS3DH_MOSI 11
// Used for hardware & software SPI
#define LIS3DH1_CS 5
#define LIS3DH2_CS 27
#define HSPI_SCK 14   // HSPI Clock
#define HSPI_MISO 12  // HSPI MISO
#define HSPI_MOSI 13  // HSPI MOSI
// hardware SPI
//Adafruit_LIS3DH lis = Adafruit_LIS3DH(LIS3DH_CS);
// Low Power 5Khz data rate needs faster SPI, and calling setPerformanceMode & setDataRate
SPIClass hspi(HSPI); // Use VSPI for hardware SPI
Adafruit_LIS3DH lis1 = Adafruit_LIS3DH(LIS3DH1_CS, &SPI, 2000000); // SPI speed 2MHz
Adafruit_LIS3DH lis2 = Adafruit_LIS3DH(LIS3DH2_CS, &hspi, 2000000); // SPI speed 2MHz

unsigned long motor_start_time = 0;
unsigned long previousMicros = 0;

bool running_test = false;

#define BUTTONPIN 26
#define MOTOR_PWM 33
#define MOTOR_IN1 32
#define motorPWMFreq 5000
#define motorPWMChannel 0
#define SAMPLE_TIME 625 // 625 microseconds = 1.6kHz
#define NUM_SAMPLES 1000 // number of samples to take

uint16_t nsample = 0;
uint16_t amplitude = 1023; // default amplitude for motor PWM


struct AccelerometerData {
  int x;
  int y;
  int z;
  unsigned long timestamp;
};
AccelerometerData data1[NUM_SAMPLES];
AccelerometerData data2[NUM_SAMPLES];


void accel_setup (Adafruit_LIS3DH &lis, lis3dh_range_t range = LIS3DH_RANGE_8_G, lis3dh_mode_t performance = LIS3DH_MODE_LOW_POWER ,lis3dh_dataRate_t dataRate = LIS3DH_DATARATE_LOWPOWER_1K6HZ) {
  if (! lis.begin()) {   // change this to 0x19 for alternative i2c address (default:0x18)
    Serial.println("Couldnt start");
    while (1) yield();
  }
  Serial.println("LIS3DH found!");

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

  // Debugging: Print raw accelerometer values
  // Serial.print("Raw X: "); Serial.print(lis.x);
  // Serial.print(" Y: "); Serial.print(lis.y);
  // Serial.print(" Z: "); Serial.println(lis.z);

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

void run_test_sequence() {
  if (digitalRead(BUTTONPIN) || running_test) {
    if (! running_test) {
      Serial.println("Button pressed, starting test sequence");
      running_test = true;
      // Start motor
      digitalWrite(MOTOR_IN1, HIGH);
      motor_start_time = millis();
      Serial.print("Motor start time: "); Serial.println(motor_start_time);
    }
    // Serial.print("Current millis: "); Serial.println(millis());
    // Serial.print("Time difference: "); Serial.println(millis() - motor_start_time);
    // Wait for 0.5 seconds to take readings
    previousMicros = micros(); // reset previousMicros to current time
    while (millis() - motor_start_time > 500) {
      // Serial.println("Motor ready, taking readings");
      unsigned long timestamp = micros();
      // sample the acceleration at 1kHz

      // debugging: Print timing values
      // Serial.print("Timestamp: "); Serial.print(timestamp);
      // Serial.print(" PreviousMillis: "); Serial.print(previousMillis);
      // Serial.print(" SAMPLE_TIME: "); Serial.println(SAMPLE_TIME);

      if (timestamp - previousMicros >= SAMPLE_TIME) {
        previousMicros += SAMPLE_TIME; // increment by sample time to avoid drift
        if (nsample < NUM_SAMPLES) {
          // Serial.println("Taking sample");
          data1[nsample] = sample_accelerometer(lis1, timestamp);
          data2[nsample] = sample_accelerometer(lis2, timestamp);
          nsample++;
        }
        else {
          Serial.println("Max samples reached, stopping test sequence");
          nsample = 0;
          Serial.println("Test sequence complete");
          // Stop motor
          digitalWrite(MOTOR_IN1, LOW);
          running_test = false;
          Serial.println("Accelerometer 1 Data:");
          print_accelerometer_data(data1, NUM_SAMPLES);
          Serial.println("Accelerometer 2 Data:");
          print_accelerometer_data(data2, NUM_SAMPLES);
          break;
        }
      }
    }
  }
}

void setup(void) {
  Serial.begin(115200);
  //while (!Serial) delay(10);     // will pause Zero, Leonardo, etc until serial console opens

  accel_setup(lis1);
  accel_setup(lis2);

  pinMode(BUTTONPIN, INPUT_PULLDOWN); // button to trigger motor
  pinMode(MOTOR_IN1, OUTPUT); // enable pin for motor

  // Set up PWM for motor
  ledcSetup(motorPWMChannel, motorPWMFreq, 10);
  ledcAttachPin(MOTOR_PWM, motorPWMChannel);
  ledcWrite(motorPWMChannel, amplitude);
  Serial.println("Setup complete");
}

void loop() {
  // listen for input from serial to set target frequency for vibration motor
  if (Serial.available() > 0) {
    amplitude = Serial.parseInt();
    if (amplitude > 1023) amplitude = 1023;
    if (amplitude < 0) amplitude = 0;
    ledcWrite(motorPWMChannel, amplitude);
    Serial.print("Amplitude: "); Serial.println(amplitude);
  }
  // Run test sequence when button is pressed
  run_test_sequence();
}