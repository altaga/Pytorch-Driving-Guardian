#include <M5Core2.h>
#include <WiFi.h>
#include <PubSubClient.h>

#define w 240
#define h 180
#define size w*h

float accX = 0.0F;  // Define variables for storing inertial sensor data
float accY = 0.0F; 
float accZ = 0.0F;
float temp = 0.0F;

const char* ssid = "YOUR_SSID";
const char* password = "YOUR_PASSWORD";
const char* mqtt_server = "192.168.xx.xx";

void setup_wifi();
void reconnect();
void callback(char* topic, byte* message, unsigned int length);

WiFiClient espClient;
PubSubClient client(espClient);

uint8_t gImage_m5_logo[size];

int counter = 0;

void setup() {
  setup_wifi();
  client.setServer(mqtt_server, 1883);
  client.setCallback(callback);
  M5.begin();
  M5.Lcd.fillScreen(GREEN);
  M5.IMU.Init();
}

void loop() {
  if (!client.connected()) {
    reconnect();
  }
  client.loop();
  M5.IMU.getAccelData(&accX, &accY, &accZ);    //Passing Acc Reference
  M5.IMU.getTempData(&temp);                   //Passing Temp Reference
  if(accX > 7.5 || accY > 7.5 || accZ > 7.5){
    client.publish("Crash","Crash");
    delay(100);
  }
}

void callback(char* topic, byte* message, unsigned int length) {
  String messageTemp="";
  if (String(topic) == "inTopic") {
  if(counter < h){
  for (int i = 0; i < length; i++) {
      gImage_m5_logo[i+(counter*w)] = (uint8_t)message[i];   
  }
  counter++;
  }
  }
  else if (String(topic) == "reset") {
    counter = 0;
    M5.Lcd.drawBitmap(40, 30, w, h, (uint8_t *)gImage_m5_logo);
  }
  else if (String(topic) == "label") {
    for (int i = 0; i < length; i++) {
    messageTemp += (char)message[i];
    }
    Serial.println(messageTemp);
    if(messageTemp == "motorbike"){
      M5.Lcd.fillRect(0, 0, 320, 30, YELLOW);
      M5.Lcd.fillRect(0, 30, 40, 210, YELLOW);
      M5.Lcd.fillRect(40, 210, 280, 30, YELLOW);
      M5.Lcd.fillRect(280, 30, 40, 180, YELLOW);
    }
    else if(messageTemp == "dog"){
      M5.Lcd.fillRect(0, 0, 320, 30, RED);
      M5.Lcd.fillRect(0, 30, 40, 210, RED);
      M5.Lcd.fillRect(40, 210, 280, 30, RED);
      M5.Lcd.fillRect(280, 30, 40, 180, RED);
    }
    else if(messageTemp == "person"){
      M5.Lcd.fillRect(0, 0, 320, 30, RED);
      M5.Lcd.fillRect(0, 30, 40, 210, RED);
      M5.Lcd.fillRect(40, 210, 280, 30, RED);
      M5.Lcd.fillRect(280, 30, 40, 180, RED);
    }
    else if(messageTemp == "car"){
      M5.Lcd.fillRect(0, 0, 320, 30, YELLOW);
      M5.Lcd.fillRect(0, 30, 40, 210, YELLOW);
      M5.Lcd.fillRect(40, 210, 280, 30, YELLOW);
      M5.Lcd.fillRect(280, 30, 40, 180, YELLOW);
    }
    else{
      M5.Lcd.fillRect(0, 0, 320, 30, GREEN);
      M5.Lcd.fillRect(0, 30, 40, 210, GREEN);
      M5.Lcd.fillRect(40, 210, 280, 30, GREEN);
      M5.Lcd.fillRect(280, 30, 40, 180, GREEN);
    }
  }
    else if (String(topic) == "labelE") {
    for (int i = 0; i < length; i++) {
    messageTemp += (char)message[i];
    }
    Serial.println(messageTemp);
    if(messageTemp == "Angry"){
      M5.Lcd.fillRect(0, 0, 320, 30, RED);
      M5.Lcd.fillRect(0, 30, 40, 210, RED);
      M5.Lcd.fillRect(40, 210, 280, 30, RED);
      M5.Lcd.fillRect(280, 30, 40, 180, RED);
    }
    else if(messageTemp == "Neutral"){
      M5.Lcd.fillRect(0, 0, 320, 30, DARKGREY);
      M5.Lcd.fillRect(0, 30, 40, 210, DARKGREY);
      M5.Lcd.fillRect(40, 210, 280, 30, DARKGREY);
      M5.Lcd.fillRect(280, 30, 40, 180, DARKGREY);
    }
    else if(messageTemp == "Happy"){
      M5.Lcd.fillRect(0, 0, 320, 30, GREEN);
      M5.Lcd.fillRect(0, 30, 40, 210, GREEN);
      M5.Lcd.fillRect(40, 210, 280, 30, GREEN);
      M5.Lcd.fillRect(280, 30, 40, 180, GREEN);
    }
    else if(messageTemp == "Sad"){
      M5.Lcd.fillRect(0, 0, 320, 30, BLUE);
      M5.Lcd.fillRect(0, 30, 40, 210, BLUE);
      M5.Lcd.fillRect(40, 210, 280, 30, BLUE);
      M5.Lcd.fillRect(280, 30, 40, 180, BLUE);
    }
    else{
      M5.Lcd.fillRect(0, 0, 320, 30, BLACK);
      M5.Lcd.fillRect(0, 30, 40, 210, BLACK);
      M5.Lcd.fillRect(40, 210, 280, 30, BLACK);
      M5.Lcd.fillRect(280, 30, 40, 180, BLACK);
    }
  }
  else if (String(topic) == "labelD") {
    for (int i = 0; i < length; i++) {
    messageTemp += (char)message[i];
    }
    Serial.println(messageTemp);
    if(messageTemp != ""){
      M5.Lcd.fillRect(0, 0, 320, 30, RED);
      M5.Lcd.fillRect(0, 30, 40, 210, RED);
      M5.Lcd.fillRect(40, 210, 280, 30, RED);
      M5.Lcd.fillRect(280, 30, 40, 180, RED);
    }
    else{
      M5.Lcd.fillRect(0, 0, 320, 30, GREEN);
      M5.Lcd.fillRect(0, 30, 40, 210, GREEN);
      M5.Lcd.fillRect(40, 210, 280, 30, GREEN);
      M5.Lcd.fillRect(280, 30, 40, 180, GREEN);
    }
  }
}

void setup_wifi() {
  delay(10);
  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
  }
}

void reconnect() {
  while (!client.connected()) {
    if (client.connect("ESP32Client")) {
      client.subscribe("inTopic");
      client.subscribe("label");
      client.subscribe("labelE");
      client.subscribe("labelD");
      client.subscribe("reset");
    } else {
      delay(5000);
    }
  }
}
