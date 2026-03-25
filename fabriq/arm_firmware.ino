#include <Adafruit_PWMServoDriver.h>

Adafruit_PWMServoDriver pwm = Adafruit_PWMServoDriver();

// Servo MIN/MAX are (us width with padding) / 1000000 * SERVO_FREQ * 4096
// Ideal PWM range is 500-2500 us
// Padded for the non-exact frequency
#define SERVO_FREQ 333                                                // 333 Hz
#define SERVO_MIN (int)(510.0f / 1000000.0f * SERVO_FREQ * 4096.0f)   // This is the 'minimum' pulse length count (out of 4096)
#define SERVO_MAX (int)(2500.0f / 1000000.0f * SERVO_FREQ * 4096.0f)  // This is the 'maximum' pulse length count (out of 4096)
#define OSCILLATOR_FREQ 26000000                                      // PWM oscillator frequency. Specific to the Adafruit board, calibrated by oscilloscope

// Oscilloscope measurements
// 338 Hz frequency
// 0 position: 501 us wiTh
// 1 position: 2499 us

#define BAUD_RATE 115200

#define PI 3.1415927f

// -------- Configuration --------
const uint32_t SYNC_1 = 0xAABBCCDD;
const uint32_t SYNC_2 = 0xBBAACCDD;
const size_t LEN_1 = 7;
const size_t LEN_2 = 8;
const uint32_t ACK_WORD = 0xDDCCBBAA;
const uint8_t NUM_ARMS = 2;
const uint8_t NUM_JOINTS = 7;

const unsigned long T = 10.0;         // Loop period in milliseconds
const float MAX_MOVEMENT_TIME = 5.0;  // Maximum movement time in seconds
const float MIN_MOVEMENT_TIME = 0.5;  // Minimum movement time in seconds


// Joint bounds in radians
const float ANGLE_BOUNDS[NUM_JOINTS][2] = {
  { -3.0f * PI / 4.0f, 3.0f * PI / 4.0f },  // -135 to 135 degrees
  { 5.0f * PI / 18.0f, PI },                // 50 to 180 degrees
  { -PI, PI / 3.0f },                       // Elbow
  { -3.0f * PI / 4.0f, 3.0f * PI / 4.0f },  // Wrist rotation
  { -PI / 2.0f, PI / 2.0f },
  { -PI / 2.0f, PI / 2.0f },
  { PI * 10.0f / 180.0f, PI * 27.0f / 180.0f },
};

const float ANGLE_BIAS[NUM_ARMS][NUM_JOINTS] = {
  { -15.0f * PI / 180.0f,  // Arm 1 (Right)
    -12.0f * PI / 180.0f,  // shoulder
    9.0f * PI / 180.0f,   // elbow (more negative, smaller angle)
    3.0f * PI / 180.0f,    // Wrist rot  (+ --> CW viewed from behind)
    2.0f * PI / 180.0f,    // Wrist catboy
    10.0f * PI / 180.0f,   // Wrist yaw
    2.0f * PI / 180.0f },
  { -18.0f * PI / 180.0f,  // Arm 0 (Left)
    -24.0f * PI / 180.0f,  // shoulder
    0.8f * PI / 180.0f,    // elbow
    10.0f * PI / 180.0f,   // Wrist rot
    -5.0f * PI / 180.0f,   // Wrist catboy
    -3.0f * PI / 180.0f,   // Wrist yaw
    0.0f * PI / 180.0f },
};

// Joint angle offsets in radians
// The '0' position of the joints is not necessarily the 0 position of the servos
const float ANGLE_OFFSETS[NUM_JOINTS] = {
  3.0f * PI / 4.0f,  // Yaw, 0 position is straight ahead, 135 degrees offset
  0,
  PI,
  3.0f * PI / 4.0f,
  PI / 2.0f,
  PI / 2.0f,
  0,
};


// Home position angle in radians (same frame as input from Python)
// Taken from Python home position
const float ANGLE_HOME[NUM_JOINTS] = {
  -0.175057,
  1.5223863,
  -2.57422625,
  -0.21336506,
  -0.96579284,
  0.12261212,
  25.0f * PI / 180.0f
};

// Everything zero/90
const float CALIBRATION_ANGLES[NUM_JOINTS] = {
  0.0f,
  PI / 2.0f,
  -PI / 2.0f,
  0.0f,
  0.0f,
  0.0f,
  25.0f * PI / 180.0f
};

// Convert radian to 0-1 PWM
// Conversion factor depends on range of motion of joint
const float ANGLE_TO_PWM[NUM_JOINTS] = {
  1.0f / (3.0f / 4.0f * 2.0f * PI),  // Yaw has 270 deg range, also multiply by the gear ratio
  1.0f / PI,                         // Shoulder
  1.0f / PI,                         // Elbow
  1.0f / (3.0f / 4.0f * 2.0f * PI),  // Wrist rotation has 270 deg range
  1.0f / PI,
  1.0f / PI,
  1.0f / PI,  // Gripper
};

// Convert joint index - 1 to PWM controller pin
// 0-6 : Right arm yaw - gripper
// 7-13 : Left arm yaw - gripper
/*
 SERVO NUM MAP
 3456 ABCD
 012  987

 PIN MAP
 4356 ABCD
 987 210 

mapping of the century
*/
const uint8_t SERVO_PINS[NUM_ARMS][NUM_JOINTS] = {
  { 9,    // Right Yaw
    8,    // Right Shoulder
    7,    // Right Elbow
    4,    // Right Wrist Rotation
    3,    // Right Wrist Flexion
    5,    // Right Wrist Abduction
    6 },   // Right Gripper
  { 0,    // Left Yaw
    1,    // Left Shoulder
    2,    // Left Elbow
    13,   // Left Wrist Rotation
    12,   // Left Wrist Flexion
    11,   // Left Wrist Abduction
    10 }, // Left Gripper
};

float receivedData[NUM_ARMS * NUM_JOINTS + 1];  // Raw angles received from host, joint frame + time. radians
float desiredAngles[NUM_ARMS][NUM_JOINTS];      // Desired angle with offsets/limits applied, motor frame. radians
float motorAngles[NUM_ARMS][NUM_JOINTS];        // Current motor angle in motor frame. radians
float startAngles[NUM_ARMS][NUM_JOINTS];        // Starting motor angle, saved whenever a new target is read from the serial buffer. radians
int interpolation_steps = 5.0f / T * 1000;      // Movement time / Sample period

float jointVel[NUM_ARMS][NUM_JOINTS];  // rad/s
float jointAcc[NUM_ARMS][NUM_JOINTS];  // rad/s^2

void setup() {
  Serial.begin(BAUD_RATE);
  pwm.begin();
  pwm.setOscillatorFrequency(OSCILLATOR_FREQ);
  pwm.setPWMFreq(SERVO_FREQ);

  // Set all servos to their home position
  for (int arm = 0; arm < NUM_ARMS; arm++) {
    for (int joint = 0; joint < NUM_JOINTS; joint++) {
      // Yaw ratio must be applied before offsets
      float ratio = joint == 0 ? 72.0f / 63.0f : 1.0f;
      motorAngles[arm][joint] = ANGLE_OFFSETS[joint] + ratio * (ANGLE_HOME[joint] + ANGLE_BIAS[arm][joint]);
      //motorAngles[arm][joint] = ANGLE_OFFSETS[joint] + (CALIBRATION_ANGLES[joint] + ANGLE_BIAS[arm][joint]);
      startAngles[arm][joint] = motorAngles[arm][joint];
      desiredAngles[arm][joint] = motorAngles[arm][joint];
      pwm.setPWM(SERVO_PINS[arm][joint], 0, pwmToPulse(motorAngles[arm][joint] * ANGLE_TO_PWM[joint]));
    }
  }
}

// Map 0-1 position to 0-4096 pulse wiTh
int pwmToPulse(float pos) {
  pos = min(max(pos, 0), 1);  // Bound 0 <= pos <= 1
  return (int)(SERVO_MIN + pos * (SERVO_MAX - SERVO_MIN));
}

// Start and end angle in radians, t is a value from 0-1, 0 representing the start, 1 representing the end
float interpolate_s(float start, float end, float t) {
  float t3 = t * t * t;
  float t4 = t3 * t;
  float t5 = t4 * t;
  float s = 6 * t5 - 15 * t4 + 10 * t3;
  return start + (end - start) * s;
}

float interpolate(float start, float end, float t) {
  float t2 = t * t;
  float t3 = t2 * t;
  float s = 0.2*t +2.4 * t2 - 1.6 * t3;
  return start + (end - start) * s;
}

float interpolate_l(float start, float end, float t) {
  float s = t;
  return start + (end - start) * s;
}

void loop() {
  static uint32_t syncBuffer = 0;
  static int interpolation_step = 0;
  unsigned long startTime = millis();

  // Read serial if its availble
  while (Serial.available()) {
    syncBuffer = (syncBuffer >> 8) | ((uint32_t)Serial.read() << 24);
    if (syncBuffer == SYNC_1) {
      size_t bytesRead = Serial.readBytes(
        (uint8_t*)receivedData,
        LEN_1 * sizeof(float));

      if (bytesRead == LEN_1 * sizeof(float)) {
        Serial.write((uint8_t*)&ACK_WORD, sizeof(ACK_WORD));
      }
      syncBuffer = 0;
    } else if (syncBuffer == SYNC_2) {
      size_t bytesRead = Serial.readBytes(
        (uint8_t*)(receivedData + LEN_1),
        LEN_2 * sizeof(float));

      if (bytesRead == LEN_2 * sizeof(float)) {
        Serial.write((uint8_t*)&ACK_WORD, sizeof(ACK_WORD));
      }
      syncBuffer = 0;

      // Put receivedData into necessary structures
      float movement_time = receivedData[NUM_ARMS * NUM_JOINTS];

      // Clamp movement time to limit the speed
      if (movement_time < MIN_MOVEMENT_TIME) {
        movement_time = MIN_MOVEMENT_TIME;
      } else if (movement_time > MAX_MOVEMENT_TIME) {
        movement_time = MAX_MOVEMENT_TIME;
      }
      interpolation_steps = movement_time / T * 1000;

      // Save the current starting angles
      for (int arm = 0; arm < NUM_ARMS; arm++) {
        for (int joint = 0; joint < NUM_JOINTS; joint++) {
          startAngles[arm][joint] = motorAngles[arm][joint];
        }
      }

      // Compute desired angles
      for (int arm = 0; arm < NUM_ARMS; arm++) {
        for (int joint = 0; joint < NUM_JOINTS; joint++) {

          // Yaw ratio must be applied before offsets
          float ratio = joint == 0 ? 72.0f / 63.0f : 1.0f;

          int i = arm * NUM_JOINTS + joint;
          // Angle check and offset application
          if (receivedData[i] <= ANGLE_BOUNDS[joint][0]) {  // min check
            desiredAngles[arm][joint] = ratio * (ANGLE_BOUNDS[joint][0] + ANGLE_BIAS[arm][joint]) + ANGLE_OFFSETS[joint];
          } else if (receivedData[i] >= ANGLE_BOUNDS[joint][1]) {  // max check
            desiredAngles[arm][joint] = ratio * (ANGLE_BOUNDS[joint][1] + ANGLE_BIAS[arm][joint]) + ANGLE_OFFSETS[joint] + ANGLE_BIAS[arm][joint];
          } else {  // within bounds
            desiredAngles[arm][joint] = ratio * (receivedData[i] + ANGLE_BIAS[arm][joint]) + ANGLE_OFFSETS[joint];
          }
        }
      }

      // Restart interpolation
      interpolation_step = 0;
    }  // if (syncBuffer == SYNC_WORD)
  }    // while (Serial.available())

  float t = (float)interpolation_step / interpolation_steps;
  if (t > 1.0) t = 1.0;

  // Set motor angles with angle and speed check
  for (int arm = 0; arm < NUM_ARMS; arm++) {
    for (int joint = 0; joint < NUM_JOINTS; joint++) {
      motorAngles[arm][joint] = interpolate(startAngles[arm][joint], desiredAngles[arm][joint], t);
      pwm.setPWM(SERVO_PINS[arm][joint], 0, pwmToPulse(motorAngles[arm][joint] * ANGLE_TO_PWM[joint]));
    }
  }

  // Update interpolation step
  if (interpolation_step < interpolation_steps) {
    interpolation_step++;
  }

  unsigned long elapsed = millis() - startTime;
  if (elapsed < T) {
    delay(T - elapsed);
  }
}
