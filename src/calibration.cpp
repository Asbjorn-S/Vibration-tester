#include <calibration.h>
#include <accelerometers.h>
#include <algorithm>
#include <wifi_mqtt.h>
#include <ArduinoJson.h>

#define DEBUG // Uncomment to enable debug prints

ArduinoFFT<double> FFT = ArduinoFFT<double>(vReal, vImag, FFT_SAMPLES, SAMPLE_FREQ); // Initialize FFT object

double vReal[FFT_SAMPLES]; // Real part of the FFT input
double vImag[FFT_SAMPLES]; // Imaginary part of the FFT input
frequencyData freqData[(maxAmplitude-minAmplitude)/CAL_STEP+2];

extern PubSubClient client; // MQTT client object

void test_frequency_v_amplitude() {
    // Placeholder for motor calibration logic
    // This function can be used to calibrate the motor based on the accelerometer data
    Serial.println("Calibrating vibration motor...");
    // Add calibration logic here
    // const uint16_t calibrationTime = 1000; // Time to measure frequency at each amplitude [ms]
  
    ledcWrite(motorPWMChannel, minAmplitude); // Set initial amplitude
    digitalWrite(MOTOR_IN1, HIGH); // Start the motor
  
    // double peakFrequency[(maxAmplitude-minAmplitude)/CAL_STEP+5] = {0};
    uint16_t frequencyIndex = 0;
  
    for (uint16_t amplitude = minAmplitude; amplitude <= maxAmplitude; amplitude += CAL_STEP) {
      // Use the last iteration to test exactly at maxAmplitude
      if (amplitude > maxAmplitude - CAL_STEP && amplitude < maxAmplitude) {
        amplitude = maxAmplitude;
      }
      // Set motor amplitude
      ledcWrite(motorPWMChannel, amplitude);
      // Wait for the motor to stabilize
      delay(200); // Wait for 200 ms to stabilize
  
      // Measure frequency
      unsigned long actualSampleStart = micros();
      uint16_t sampleIndex = 0;
            while (sampleIndex < FFT_SAMPLES) {
        AccelerometerData sample = sample_accelerometer(lis1);
  
        // Store the accelerometer's X-axis data in the FFT input array
        vReal[sampleIndex] = sample.x; // Magnitude of acceleration vector
        vImag[sampleIndex] = 0; // Imaginary part is zero for real input 
        sampleIndex++;
  
        // Wait for next sample
        delayMicroseconds(SAMPLE_TIME);
      }
      
      // Calculate actual sampling frequency (for debugging)
      unsigned long actualSampleEnd = micros();
      float actualSampleTime = (actualSampleEnd - actualSampleStart) / (float)FFT_SAMPLES;
      float actualSampleFreq = 1000000.0 / actualSampleTime;
      
      #ifdef DEBUG
      Serial.print("Actual sampling time: ");
      Serial.print(actualSampleTime);
      Serial.println(" µs");
      Serial.print("Actual sampling frequency: ");
      Serial.print(actualSampleFreq);
      Serial.println(" Hz");
      #endif
      
            // Perform FFT
      FFT.windowing(FFT_WIN_TYP_HAMMING, FFT_FORWARD); // Apply a Hamming window
      FFT.compute(FFT_FORWARD); // Compute the FFT
      FFT.complexToMagnitude(); // Compute magnitudes

      double maxMagnitude = 0;
      uint16_t peakIndex = 0;
      
      // Only search up to Nyquist frequency (half of sampling frequency)
      for (uint16_t i = 1; i < (FFT_SAMPLES / 2); i++) {
        if (vReal[i] > maxMagnitude) {
          maxMagnitude = vReal[i];
          peakIndex = i;
        }
      }
      
      // Calculate the exact frequency using the peak bin and ACTUAL sampling frequency
      double peakFrequency = peakIndex * (actualSampleFreq / FFT_SAMPLES);
      
      // Interpolate peak for higher accuracy
      if (peakIndex > 0 && peakIndex < (FFT_SAMPLES/2 - 1)) {
        double leftVal = vReal[peakIndex - 1];
        double midVal = vReal[peakIndex];
        double rightVal = vReal[peakIndex + 1];
        
        // Parabolic interpolation formula
        double delta = 0.5 * (leftVal - rightVal) / (leftVal - 2*midVal + rightVal);
        
        // Apply correction only if reasonable
        if (!isnan(delta) && abs(delta) < 1.0) {
          double interpPeakIndex = peakIndex + delta;
          peakFrequency = interpPeakIndex * (actualSampleFreq / FFT_SAMPLES);
        }
      }
      
      #ifdef DEBUG
      // Print the magnitude spectrum for debugging
      Serial.println("FFT Magnitude Spectrum:");
      for (uint16_t i = 0; i < 20; i++) {
        double freq = i * (actualSampleFreq / FFT_SAMPLES);
        Serial.print("Bin ");
        Serial.print(i);
        Serial.print(" (");
        Serial.print(freq);
        Serial.print(" Hz): ");
        Serial.println(vReal[i]);
      }
      Serial.print("Peak at bin ");
      Serial.print(peakIndex);
      Serial.print(" (");
      Serial.print(peakFrequency);
      Serial.println(" Hz)");
      #endif
      
      // Store the frequency in the data structure
      freqData[frequencyIndex].frequency = peakFrequency;
      freqData[frequencyIndex].amplitude = amplitude;
      
      #ifdef DEBUG
      Serial.print("Amplitude: ");
      Serial.print(amplitude);
      Serial.print(" -> Detected frequency: ");
      Serial.print(peakFrequency);
      Serial.println(" Hz");
      #endif
      
      frequencyIndex++;
    }
    // Turn off the motor after calibration
    digitalWrite(MOTOR_IN1, LOW);
    ledcWrite(motorPWMChannel, minAmplitude); // Set amplitude to minimum 
    // Serial.println("Calibration complete.");
    calibrationComplete = true;
  }

  //publish the frequency data to MQTT broker
  void publishCalibrationData() {
    if (!calibrationComplete) {
      Serial.println("Cannot publish: Calibration not complete");
      return;
    }
    
    const uint16_t numDataPoints = ((maxAmplitude - minAmplitude) / CAL_STEP) + 1;
    
    // Publish overall calibration status
    client.publish("vibration/calibration/status", "complete");
    
    // Create and publish JSON object with all calibration points
    DynamicJsonDocument doc(1024); // Adjust size as needed
    JsonArray dataArray = doc.createNestedArray("calibration_data");
    
    for (uint16_t i = 0; i < numDataPoints; i++) {
      JsonObject dataPoint = dataArray.createNestedObject();
      dataPoint["amplitude"] = freqData[i].amplitude;
      dataPoint["frequency"] = freqData[i].frequency;
    }
    
    // Convert to JSON string
    String jsonOutput;
    size_t jsonSize = serializeJson(doc, jsonOutput);
    
    // Publish full dataset
    client.publish("vibration/calibration/data", jsonOutput.c_str());
    
    Serial.print("Calibration data published to MQTT: ");Serial.println(jsonSize);

  }
  
  uint16_t frequencyToAmplitude(float targetFrequency) {
    // Number of calibration data points
    const uint16_t numDataPoints = ((maxAmplitude - minAmplitude) / CAL_STEP) + 1;
    
    // Handle out-of-range cases
    if (targetFrequency <= 0) {
      return 0; // Motor off
    }
    
    // If target is lower than our minimum frequency, return minimum amplitude
    if (targetFrequency < freqData[0].frequency) {
      return freqData[0].amplitude;
    }
    
    // If target is higher than our maximum frequency, return maximum amplitude
    if (targetFrequency > freqData[numDataPoints - 1].frequency) {
      return freqData[numDataPoints - 1].amplitude;
    }
    
    // Find the two closest frequency data points
    uint16_t lowerIndex = 0;
    uint16_t upperIndex = 0;
    
    for (uint16_t i = 0; i < numDataPoints - 1; i++) {
      if (freqData[i].frequency <= targetFrequency && 
          freqData[i + 1].frequency >= targetFrequency) {
        lowerIndex = i;
        upperIndex = i + 1;
        break;
      }
    }
    
    // Perform linear interpolation between the two closest points
    float lowerFreq = freqData[lowerIndex].frequency;
    float upperFreq = freqData[upperIndex].frequency;
    uint16_t lowerAmp = freqData[lowerIndex].amplitude;
    uint16_t upperAmp = freqData[upperIndex].amplitude;
    
    // Calculate the interpolated amplitude (linear interpolation)
    float ratio = (targetFrequency - lowerFreq) / (upperFreq - lowerFreq);
    uint16_t interpolatedAmplitude = lowerAmp + ratio * (upperAmp - lowerAmp);
    
    // Ensure the result is within valid range
    if (interpolatedAmplitude > maxAmplitude) {
      interpolatedAmplitude = maxAmplitude;
    }
    if (interpolatedAmplitude < minAmplitude) {
      interpolatedAmplitude = minAmplitude;
    }
    
    #ifdef DEBUG
      Serial.print("Target frequency: ");
      Serial.print(targetFrequency);
      Serial.print(" Hz -> Mapped amplitude: ");
      Serial.println(interpolatedAmplitude);
    #endif
    
    return interpolatedAmplitude;
  }