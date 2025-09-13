#include "FastIMU.h"
#include <Wire.h>

#define IMU_ADDRESS 0x6B        // Change to the address of the IMU
#define PERFORM_CALIBRATION      // Comment to disable startup calibration

QMI8658 IMU;                     // Keep the same IMU type as your original

// Currently supported IMUS: MPU9255 MPU9250 MPU6886 MPU6500 MPU6050 ICM20689 ICM20690 BMI055 BMX055 BMI160 LSM6DS3 LSM6DSL QMI8658

calData calib = { 0 };  // Calibration data
AccelData accelData;    // Sensor data
GyroData  gyroData;
MagData   magData;

// ===== Gesture logging config (minimal additions) =====
static const uint16_t NUM_SAMPLES        = 100;   // ~2 s @ 50 Hz
static const uint16_t SAMPLE_PERIOD_MS   = 20;    // 50 Hz
static const float    ACC_TRIGGER_G      = 0.20f; // motion start threshold
static const uint16_t NUM_REPS_PER_CLASS = 30;

static volatile uint8_t current_label    = 1;     // 1 or 2
static volatile bool    request_one_burst = false;
static volatile bool    request_30_reps   = false;
static bool             is_capturing      = false;

// --- tiny helpers (additions) ---
void print_help() {
  Serial.println(F("\n=== Gesture Logger (Serial) ==="));
  Serial.println(F("1 -> set label to 1"));
  Serial.println(F("2 -> set label to 2"));
  Serial.println(F("b -> record ONE burst"));
  Serial.println(F("r -> record 30 bursts"));
  Serial.println(F("h -> help"));
}

// Wait until acceleration deviates from a short baseline by ACC_TRIGGER_G
bool wait_for_motion(uint32_t timeout_ms = 10000) {
  // Baseline
  float ax0 = 0, ay0 = 0, az0 = 0;
  int   n   = 0;
  uint32_t t0 = millis();

  while (n < 20 && millis() - t0 < 1000) {
    IMU.update();
    IMU.getAccel(&accelData);
    ax0 += accelData.accelX;
    ay0 += accelData.accelY;
    az0 += accelData.accelZ;
    n++;
    delay(5);
  }
  if (n == 0) return false;
  ax0 /= n; ay0 /= n; az0 /= n;

  // Wait for deviation
  t0 = millis();
  while (millis() - t0 < timeout_ms) {
    IMU.update();
    IMU.getAccel(&accelData);
    float dx = fabsf(accelData.accelX - ax0)
             + fabsf(accelData.accelY - ay0)
             + fabsf(accelData.accelZ - az0);
    if (dx > ACC_TRIGGER_G) return true;
    delay(5);
  }
  return false;
}

// Capture one window and print CSV lines
bool capture_burst_stream(uint8_t label, uint16_t rep) {
  if (!wait_for_motion(10000)) {
    Serial.println("# Timeout waiting motion");
    return false;
  }

  uint32_t t_start = millis();
  for (uint16_t i = 0; i < NUM_SAMPLES; i++) {
    IMU.update();
    IMU.getAccel(&accelData);
    IMU.getGyro(&gyroData);

    uint32_t t_ms = millis() - t_start;

    // CSV: label,rep,idx,t_ms,ax,ay,az,gx,gy,gz
    Serial.printf("%u,%u,%u,%u,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n",
                  label, rep, i, t_ms,
                  accelData.accelX, accelData.accelY, accelData.accelZ,
                  gyroData.gyroX,   gyroData.gyroY,   gyroData.gyroZ);

    // Pace to 50 Hz
    uint32_t t_next = t_start + (i + 1) * SAMPLE_PERIOD_MS;
    int32_t  dt     = (int32_t)t_next - (int32_t)millis();
    if (dt > 0) delay(dt);
  }
  return true;
}

void setup() {
  Wire.begin(48, 47);
  Wire.setClock(400000); // 400 kHz clock
  Serial.begin(115200);
  while (!Serial) { ; }

  int err = IMU.init(calib, IMU_ADDRESS);
  if (err != 0) {
    Serial.print("Error initializing IMU: ");
    Serial.println(err);
    while (true) { ; }
  }

#ifdef PERFORM_CALIBRATION
  Serial.println("FastIMU calibration & data example");
  if (IMU.hasMagnetometer()) {
    delay(1000);
    Serial.println("Move IMU in figure 8 pattern until done.");
    delay(3000);
    IMU.calibrateMag(&calib);
    Serial.println("Magnetic calibration done!");
  }
  else {
    delay(5000);
  }

  delay(5000);
  Serial.println("Keep IMU level.");
  delay(5000);
  IMU.calibrateAccelGyro(&calib);
  Serial.println("Calibration done!");
  Serial.println("Accel biases X/Y/Z: ");
  Serial.print(calib.accelBias[0]); Serial.print(", ");
  Serial.print(calib.accelBias[1]); Serial.print(", ");
  Serial.println(calib.accelBias[2]);
  Serial.println("Gyro biases X/Y/Z: ");
  Serial.print(calib.gyroBias[0]); Serial.print(", ");
  Serial.print(calib.gyroBias[1]); Serial.print(", ");
  Serial.println(calib.gyroBias[2]);
  if (IMU.hasMagnetometer()) {
    Serial.println("Mag biases X/Y/Z: ");
    Serial.print(calib.magBias[0]); Serial.print(", ");
    Serial.print(calib.magBias[1]); Serial.print(", ");
    Serial.println(calib.magBias[2]);
    Serial.println("Mag Scale X/Y/Z: ");
    Serial.print(calib.magScale[0]); Serial.print(", ");
    Serial.print(calib.magScale[1]); Serial.print(", ");
    Serial.println(calib.magScale[2]);
  }
  delay(5000);
  IMU.init(calib, IMU_ADDRESS);
#endif

  // err = IMU.setGyroRange(500);
  // err = IMU.setAccelRange(2);
  if (err != 0) {
    Serial.print("Error Setting range: ");
    Serial.println(err);
    while (true) { ; }
  }

  print_help();
}

void loop() {
  // --- Minimal serial UI for gesture logging (added) ---
  if (Serial.available()) {
    char c = Serial.read();
    if (c == '1') { current_label = 1; Serial.println("# Label=1"); }
    else if (c == '2') { current_label = 2; Serial.println("# Label=2"); }
    else if (c == 'b') { request_one_burst = true; }
    else if (c == 'r') { request_30_reps   = true; }
    else if (c == 'h' || c == '?') { print_help(); }
  }

  // --- Record ONE burst (CSV) ---
  if (request_one_burst && !is_capturing) {
    is_capturing = true;
    request_one_burst = false;

    Serial.println("###CSV_BEGIN");
    Serial.println("label,rep,idx,t_ms,ax,ay,az,gx,gy,gz");
    bool ok = capture_burst_stream(current_label, 0);
    Serial.println("###CSV_END");
    if (!ok) Serial.println("# No motion / failed");

    is_capturing = false;
  }

  // --- Record 30 bursts (CSV) ---
  if (request_30_reps && !is_capturing) {
    is_capturing = true;
    request_30_reps = false;

    Serial.println("###CSV_BEGIN");
    Serial.println("label,rep,idx,t_ms,ax,ay,az,gx,gy,gz");

    for (uint16_t rep = 1; rep <= NUM_REPS_PER_CLASS; rep++) {
      //Serial.printf("# Recording rep %u/%u (label=%u)\n", rep, NUM_REPS_PER_CLASS, current_label);
      if (!capture_burst_stream(current_label, rep)) {
        Serial.println("# Timeout. Re-run 'r' when ready.");
        break;
      }
      delay(300);
    }

    Serial.println("###CSV_END");
    is_capturing = false;
  }

}
