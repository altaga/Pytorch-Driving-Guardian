import cv2
import torch.hub
import os
import utils.model
from PIL import Image
from torchvision import transforms
from utils.grad_cam import BackPropagation
import time 
import threading
import vlc
import json
import os
import paho.mqtt.client as paho
import socket
import numpy as np

current_dir = os.path.dirname(os.path.realpath(__file__))
mqttc = paho.Client()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Alarm sound file
file = 'alarm.mp3'
# Sound player start
p = vlc.MediaPlayer(current_dir+'/'+file)

timebasedrow= time.time()
timebasedis= time.time()
timerundrow= time.time()
timerundis= time.time()

face_cascade = cv2.CascadeClassifier(current_dir+'/haar_models/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(current_dir+'/haar_models/haarcascade_eye.xml') 
MyModel="BlinkModel.t7"

shape = (24,24)
classes = [
    'Close',
    'Open',
]

eyess=[]
cface=0
state=""

def RGB888toRGB565(img):
    img = img.astype(np.uint8)
    temp = (img[:, :, 0]>>5) << 5 | (img[:, :, 1]>>5) << 2 | (img[:, :, 2]>>6)
    return temp

def extract_ip():
    st = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:       
        st.connect(('10.255.255.255', 1))
        IP = st.getsockname()[0]
    except Exception:
        IP = '127.0.0.1'
    finally:
        st.close()
    return IP

def on_connect(client, userdata, flags, rc):
    print("Connection returned result: " + str(rc) )
    #client.subscribe("#" , 1 ) # Wild Card

# This function trigger every time we receive a message from the platform
def on_message(client, userdata, msg):
    print("topic: "+msg.topic)
    print("payload: "+str(msg.payload))
    
# This function trigger when we subscribe to a new topic  
def on_subscribe(client, obj, mid, granted_qos):
    print("Subscribed: " + str(mid) + " " + str(granted_qos))

mqttc.on_connect = on_connect
mqttc.on_message = on_message
mqttc.on_subscribe = on_subscribe
mqttc.connect(extract_ip(), 1883, keepalive=60)
mqttc.subscribe("Crash")
        
def preprocess(image_path):
    global cface
    transform_test = transforms.Compose([
        transforms.ToTensor()
    ])
    image = cv2.imread(image_path['path'])    
    faces = face_cascade.detectMultiScale(
        image,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(1, 1),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    if len(faces) == 0:
        ...
    else:
        cface=1
        (x, y, w, h) = faces[0]
        face = image[y:y + h, x:x + w]
        cv2.rectangle(image,(x,y),(x+w,y+h),(255,255,0),2)
        roi_color = image[y:y+h, x:x+w]
        """
        Depending on the quality of your camera, this number can vary 
        between 10 and 40, since this is the "sensitivity" to detect the eyes.
        """
        sensi=20
        eyes = eye_cascade.detectMultiScale(face,1.3, sensi) 
        i=0
        for (ex,ey,ew,eh) in eyes:
            (x, y, w, h) = eyes[i]
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
            eye = face[y:y + h, x:x + w]
            eye = cv2.resize(eye, shape)
            eyess.append([transform_test(Image.fromarray(eye).convert('L')), eye, cv2.resize(face, (48,48))])
            i=i+1
    cv2.imwrite(current_dir+'/temp-images/display.jpg',image) 
    

def eye_status(image, name, net):
    img = torch.stack([image[name]])
    bp = BackPropagation(model=net)
    probs, ids = bp.forward(img)
    actual_status = ids[:, 0]
    prob = probs.data[:, 0]
    if actual_status == 0:
        prob = probs.data[:,1]

    #print(name,classes[actual_status.data], probs.data[:,0] * 100)
    return classes[actual_status.data]

def func(imag,modl):
    drow(images=[{'path': imag, 'eye': (0,0,0,0)}],model_name=modl)

def drow(images, model_name):
    global eyess
    global cface
    global timebasedrow
    global timebasedis
    global timerundrow
    global timerundis
    global state
    net = model.Model(num_classes=len(classes))
    checkpoint = torch.load(os.path.join(current_dir+'/model', model_name), map_location=device)
    net.load_state_dict(checkpoint['net'])
    net.eval()
    flag =1
    status=""
    for i, image in enumerate(images):
        if(flag):
            preprocess(image)
            flag=0
        if cface==0:
            image = cv2.imread(current_dir+"/temp-images/display.jpg")
            image = cv2.putText(image, 'No face Detected', (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 6, (0, 0, 0), 20, cv2.LINE_AA)
            image = cv2.putText(image, 'No face Detected', (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 6, (255, 255, 255), 10, cv2.LINE_AA)
            cv2.imwrite(current_dir+'/temp-images/display.jpg',image)
            timebasedrow= time.time()
            timebasedis= time.time()
            timerundrow= time.time()
            timerundis= time.time()
        elif(len(eyess)!=0):
            eye, eye_raw , face = eyess[i]
            image['eye'] = eye
            image['raw'] = eye_raw
            image['face'] = face
            timebasedrow= time.time()
            timerundrow= time.time()
            for index, image in enumerate(images):
                status = eye_status(image, 'eye', net)
                if(status =="Close"):
                    timerundis= time.time()
                    if((timerundis-timebasedis)>1.5):
                        image = cv2.imread(current_dir+'/temp-images/display.jpg')
                        state="Distracted"
                        image = cv2.putText(image, 'Distracted', (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 7, (0, 0, 0), 20, cv2.LINE_AA)
                        image = cv2.putText(image, 'Distracted', (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 7, (255, 255, 255), 10, cv2.LINE_AA)
                        cv2.imwrite(current_dir+'/temp-images/display.jpg',image)
                        if(not(p.is_playing())):
                            p.play()
                else:
                    p.stop()        
        else:
            timerundrow= time.time()
            if((timerundrow-timebasedrow)>3):
                if(not(p.is_playing())):
                    p.play()
                image = cv2.imread(current_dir+'/temp-images/display.jpg')
                state="Drowsy"
                image = cv2.putText(image, 'Drowsy', (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 7, (0, 0, 0), 20, cv2.LINE_AA)
                image = cv2.putText(image, 'Drowsy', (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 7, (255, 255, 255), 10, cv2.LINE_AA)
                cv2.imwrite(current_dir+'/temp-images/display.jpg',image)

video_capture = cv2.VideoCapture(0)

try:
    while 1:
        mqttc.loop() 
        eyess=[]
        cface=0
        state = ""
        ret, img = video_capture.read() 
        cv2.imwrite(current_dir+'/temp-images/img.jpg',img) 
        func(current_dir+'/temp-images/img.jpg',MyModel)
        img = cv2.imread(current_dir+'/temp-images/display.jpg')
        w = 240
        h = 180
        dim = (w, h)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
        img = RGB888toRGB565(img)
        img = img.flatten()
        for x in range(0,h):
            mqttc.publish("inTopic", bytearray(img[x*w:x*w+w]))
            time.sleep(0.02)
        time.sleep(0.1)
        mqttc.publish("labelD",state)
        time.sleep(0.1)
        mqttc.publish("reset","reset")

except:
    print("\nKeyboard Interrupt")
    mqttc.disconnect()
    video_capture.release()
    print("\nGoodbye")