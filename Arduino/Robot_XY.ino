#include <InverseK.h>
#include <Braccio.h>
#include <Servo.h>
const byte numChars = 32;
char receivedChars[numChars];
char tempChars[numChars];        // temporary array for use when parsing

// variables to hold the parsed data
char messageFromPC[numChars] = {0};

float x = 0;
float y = 0;
float z = 0;
float m1, m2, m3, m4;
float a0, a1, a2, a3;

boolean newData = false;
boolean received = false;

int i = 2;
Servo base;
Servo shoulder;
Servo elbow;
Servo wrist_rot;
Servo wrist_ver;
Servo gripper;

void setup() {
  Serial.begin(115200);
  Braccio.begin();
  Link base, upperarm, forearm, hand;

  base.init(74, b2a(0.0), b2a(180.0));
  upperarm.init(125, b2a(15.0), b2a(165.0));
  forearm.init(125, b2a(0.0), b2a(180.0));
  hand.init(160, b2a(0.0), b2a(180.0));

  // Attach the links to the inverse kinematic model
  InverseK.attach(base, upperarm, forearm, hand);

}

void loop() {
  recvWithStartEndMarkers();
  if (newData == true) {
    strcpy(tempChars, receivedChars);
    parseData();
    invK();
    Move();
    //    showParsedData();
    newData = false;
  }
}

//Braccio.ServoMovement(30, a2b(a0) - 15, a2b(a1), a2b(a2), 180 - a2b(a3), 90,  100);
void invK() {
  if (InverseK.solve(x, y, z, a0, a1, a2, a3)) {
    Serial.print("x: "); Serial.println(x);
    Serial.print("y: "); Serial.println(y);
    Serial.print("z: "); Serial.println(z);
//    Serial.print(a2b(a0) - 15); Serial.print(',');
//    Serial.print(a2b(a1)); Serial.print(',');
//    Serial.print(a2b(a2)); Serial.print(',');
//    Serial.println(180 - a2b(a3));
    m1 = a2b(a0) - 15; m2 =  a2b(a1); m3 = a2b(a2); m4 = 180 - a2b(a3);
  } else {
    Serial.println("No solution found!");
  }
}

//play
void Move() {
  PickTray(i);
  Braccio.ServoMovement(20, m1, m2+30, m3, m4, 90, 75);
  Braccio.ServoMovement(30, m1,    m2, m3, m4, 90, 75);
  delay(1000);
  Braccio.ServoMovement(30, m1,    m2, m3, m4, 90, 100);
  delay(500);
  Braccio.ServoMovement(20, m1, m2+30, m3, m4, 90, 75);
  Braccio.ServoMovement(20, 170, 140,   0,  180, 90,   100); //Ready!!
  i++;
  if (i > 5) {
    i = 2;
  }
}

//PickTray
void PickTray(int i){
     if (i == 1){
      Braccio.ServoMovement(20, 140,  140, 0, 180,  90,  100); //หมุน
      delay(1000);
      Braccio.ServoMovement(20, 193,  140, 0, 180,  90,  100); //หมุน
      delay(1000);
      Braccio.ServoMovement(20, 193,  88, 0, 170,  90,   100); //หยิบจากถาด1
      delay(1000);
      Braccio.ServoMovement(20, 193,  88, 0, 170,  90,   75); //หยิบ
      delay(1000);
      Braccio.ServoMovement(20, 193,  100, 0, 170,  90,   75); //ยก
      delay(1000);
    }
    else if (i == 2){
      Braccio.ServoMovement(20, 180,  140, 0, 180,  90,  100); //หมุน
      delay(1000);
      Braccio.ServoMovement(20, 180,  56, 20, 149,  90,   100); //หยิบจากถาด2
      delay(1000);
      Braccio.ServoMovement(20, 180,  56, 20, 149,  90,   75); //หยิบ
      delay(1000);
      Braccio.ServoMovement(20, 180,  80, 20, 149,  90,   75); //ยก
      delay(1000);
    }
    else if (i == 3){
      Braccio.ServoMovement(20, 158,  140, 0, 180,  90,  100); //หมุน
      delay(1000);
      Braccio.ServoMovement(20, 158,  74, 0, 163,  90,   100); //หยิบจากถาด2
      delay(1000);
      Braccio.ServoMovement(20, 158,  74, 0, 163,  90,   75); //หยิบ
      delay(1000);
      Braccio.ServoMovement(20, 158,  100, 0, 163,  90,   75); //ยก
      delay(1000);
    }
    else if (i == 4){
      Braccio.ServoMovement(20, 160,  140, 0, 180,  90,  100); //หมุน
      delay(1000);
      Braccio.ServoMovement(20, 160,  57, 20, 147,  90,   100); //หยิบจากถาด2
      delay(1000);
      Braccio.ServoMovement(20, 160,  57, 20, 147,  90,   75); //หยิบ
      delay(1000);
      Braccio.ServoMovement(20, 160,  80, 20, 147,  90,   75); //ยก
      delay(1000);
    }
    else if (i == 5){
      Braccio.ServoMovement(20, 143,  140, 0, 180,  90,  100); //หมุน
      delay(1000);
      Braccio.ServoMovement(20, 143,  49, 30, 150,  90,   100); //หยิบจากถาด2
      delay(1000);
      Braccio.ServoMovement(20, 143,  49, 30, 150,  90,   75); //หยิบ
      delay(1000);
      Braccio.ServoMovement(20, 143,  80, 30, 150,  90,   75); //ยก
      delay(1000);
    }
  }

void recvWithStartEndMarkers() {
  static boolean recvInProgress = false;
  static byte ndx = 0;
  char startMarker = '<';
  char endMarker = '>';
  char rc;

  while (Serial.available() > 0 && newData == false) {
    rc = Serial.read();
    if (recvInProgress == true) {
      if (rc != endMarker) {
        receivedChars[ndx] = rc;
        ndx++;
        if (ndx >= numChars) {
          ndx = numChars - 1;
        }
      }
      else {
        receivedChars[ndx] = '\0'; // terminate the string
        recvInProgress = false;
        ndx = 0;
        newData = true;

      }
    }

    else if (rc == startMarker) {
      recvInProgress = true;
      received = true;
    }
  }
}

void parseData() {      // split the data into its parts

  char * strtokIndx; // this is used by strtok() as an index

  strtokIndx = strtok(tempChars, ",");     // get the first part - the string
  x = atof(strtokIndx);     // convert this part to an integer

  strtokIndx = strtok(NULL, ",");
  y = atof(strtokIndx);     // convert this part to an float

  strtokIndx = strtok(NULL, ",");
  z = atof(strtokIndx);     // convert this part to an float

  //    strtokIndx = strtok(tempChars,",");      // get the first part - the string
  //    strcpy(messageFromPC, strtokIndx); // copy it to messageFromPC

}

void showParsedData() {
  //    Serial.print("Message ");
  //    Serial.println(messageFromPC);
//  Serial.print("x: ");
//  Serial.println(x);
//  Serial.print("y: ");
//  Serial.println(y);
//  Serial.print("z: ");
//  Serial.println(z);
  }

float b2a(float b) {
  return b / 180.0 * PI - HALF_PI;
}
float a2b(float a) {
  return (a + HALF_PI) * 180 / PI;
}
