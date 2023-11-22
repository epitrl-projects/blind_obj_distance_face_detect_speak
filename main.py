import os
import cv2
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import train

import re
import RPi.GPIO as GPIO 

import time 
import threading

import pyttsx3
from subprocess import call

speaktext = ""
speakbool = False

def speak():
    global speakbool,speaktext
    while True:
        if(speakbool):
            print(speaktext)
            # os.system('espeak -s 200 "${speaktext}"')
            call(['espeak "'+speaktext+'" 2>/dev/null'], shell=True)

            

            # engine = pyttsx3.init()
            speakbool = False
            # engine.say(speaktext)
            # engine.runAndWait()

distancedata = -1

GPIO.setwarnings(False) 
GPIO.setmode(GPIO.BCM) 

TRIG = 18  # Trigger pin of the Ultrasonic Sensor 
ECHO = 23 #Echo pin of the Ultrasonic Sensor 
GPIO.setup(TRIG,GPIO.OUT) 
GPIO.setup(ECHO,GPIO.IN) 


def measure(): 
    global distancedata
    while True:
        dist1=250 
        GPIO.output(TRIG, True) 
        time.sleep(0.00001) 
        GPIO.output(TRIG, False) 
        echo_state=0 
        
        while echo_state == 0: 
            echo_state = GPIO.input(ECHO) 
            pulse_start = time.time() 
        while GPIO.input(ECHO)==1: 
            pulse_end = time.time() 
        pulse_duration = pulse_end - pulse_start 
        distance = pulse_duration * 17150 
        distance = round(distance, 2) 
        if(distance<250 and distance>5): #To filter out junk values 
            dist1=distance 
        else:
            dist1 = 0
        distancedata = dist1
        print(distancedata)
        time.sleep(1)


statevars = ["T", "R"]

# Get the name of the person
name = input('Enter Procedure: ')

if (not name.isalpha() and name not in statevars):
    print("Please enter a valid Procedure")
    exit()


if (name == "T"):
    train.train()





datalabels={}
def prepare_dataset(path):
    count = 0
    labels, faces = [], []
    for person in os.listdir(path):
        person_path = os.path.join(path, person)
        if os.path.isdir(person_path):
            for img in os.listdir(person_path):
                img_path = os.path.join(person_path, img)
                face = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                face = cv2.resize(face, (100, 100))

                faces.append(face)
                count = count + 1
                datalabels[count] = img.split('.')[0]
                labels.append(count)
    return np.array(faces), np.array(labels)

def train_models(faces, labels):
    lbph = cv2.face.LBPHFaceRecognizer_create()
    eigen = cv2.face.EigenFaceRecognizer_create()
    lbph.train(faces, labels)
    eigen.train(faces, labels)
    return lbph, eigen

def save_models(lbph, eigen):
    lbph.write('lbph_model.yml')
    eigen.write('eigen_model.yml')

def load_models():
    lbph = cv2.face.LBPHFaceRecognizer_create()
    eigen = cv2.face.EigenFaceRecognizer_create()
    lbph.read('lbph_model.yml')
    eigen.read('eigen_model.yml')
    return lbph, eigen

def evaluate_models(lbph, eigen, faces, labels):
    lbph_preds, eigen_preds = [], []
    for face in faces:
        _1, lbph_pred = lbph.predict(face)
        _, eigen_pred = eigen.predict(face)
        cv2.putText(face, str(_1), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1.0, (0, 255, 0), 2)
        cv2.putText(face, str(_), (80, 50), cv2.FONT_HERSHEY_COMPLEX, 1.0, (0, 255, 0), 2)
        cv2.imshow('face', face)
        lbph_preds.append(_1)
        eigen_preds.append(_)
        print(_1,_, lbph_pred, eigen_pred)

    lbph_accuracy = accuracy_score(labels, lbph_preds)
    lbph_precision = precision_score(labels, lbph_preds, average='weighted')
    lbph_recall = recall_score(labels, lbph_preds, average='weighted')
    lbph_f1 = f1_score(labels, lbph_preds, average='weighted')

    eigen_accuracy = accuracy_score(labels, eigen_preds)
    eigen_precision = precision_score(labels, eigen_preds, average='weighted')
    eigen_recall = recall_score(labels, eigen_preds, average='weighted')
    eigen_f1 = f1_score(labels, eigen_preds, average='weighted')
    
    return lbph_accuracy, lbph_precision, lbph_recall, lbph_f1,eigen_accuracy,eigen_precision,eigen_recall,eigen_f1



def draw_label(image, point, label, font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=1, thickness=2, colour=(0, 255, 0)):
    size = cv2.getTextSize(label, font, font_scale, thickness)[0]
    x, y = point
    cv2.rectangle(image, (x, y-size[1]), (x+size[0], y), colour, cv2.FILLED)
    cv2.putText(image, label, point, font, font_scale,
                (255, 255, 255), thickness)


cap = cv2.VideoCapture(0)


import os

import cv2
import numpy as np
import sys
import time
from threading import Thread
import importlib.util

MODEL_NAME = "Sample_TFLite_model"
GRAPH_NAME = "detect.tflite"
LABELMAP_NAME = "labelmap.txt"
min_conf_threshold = float(0.5)
resW, resH = "1280x720".split('x')
imW, imH = int(resW), int(resH)
use_TPU = False

cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
cap.set(3,640)
cap.set(4,480)
# Import TensorFlow libraries
# If tflite_runtime is installed, import interpreter from tflite_runtime, else import from regular tensorflow
# If using Coral Edge TPU, import the load_delegate library
pkg = importlib.util.find_spec('tflite_runtime')
if pkg:
    from tflite_runtime.interpreter import Interpreter
    if use_TPU:
        from tflite_runtime.interpreter import load_delegate
else:
    from tensorflow.lite.python.interpreter import Interpreter
    if use_TPU:
        from tensorflow.lite.python.interpreter import load_delegate

# If using Edge TPU, assign filename for Edge TPU model
if use_TPU:
    # If user has specified the name of the .tflite file, use that name, otherwise use default 'edgetpu.tflite'
    if (GRAPH_NAME == 'detect.tflite'):
        GRAPH_NAME = 'edgetpu.tflite'       

# Get path to current working directory
CWD_PATH = os.getcwd()

# Path to .tflite file, which contains the model that is used for object detection
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,GRAPH_NAME)

# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH,MODEL_NAME,LABELMAP_NAME)

# Load the label map
with open(PATH_TO_LABELS, 'r') as f:
    labelsdect = [line.strip() for line in f.readlines()]

# Have to do a weird fix for label map if using the COCO "starter model" from
# https://www.tensorflow.org/lite/models/object_detection/overview
# First label is '???', which has to be removed.
if labelsdect[0] == '???':
    del(labelsdect[0])

# Load the Tensorflow Lite model.
# If using Edge TPU, use special load_delegate argument
if use_TPU:
    interpreter = Interpreter(model_path=PATH_TO_CKPT,
                              experimental_delegates=[load_delegate('libedgetpu.so.1.0')])
    print(PATH_TO_CKPT)
else:
    interpreter = Interpreter(model_path=PATH_TO_CKPT)

interpreter.allocate_tensors()

# Get model details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]

floating_model = (input_details[0]['dtype'] == np.float32)

input_mean = 127.5
input_std = 127.5

# Check output layer name to determine if this model was created with TF2 or TF1,
# because outputs are ordered differently for TF2 and TF1 models
outname = output_details[0]['name']

if ('StatefulPartitionedCall' in outname): # This is a TF2 model
    boxes_idx, classes_idx, scores_idx = 1, 3, 0
else: # This is a TF1 model
    boxes_idx, classes_idx, scores_idx = 0, 1, 2

# Initialize frame rate calculation
frame_rate_calc = 1
freq = cv2.getTickFrequency()


def main():
    global frame_rate_calc,freq,output_details,input_details,speaktext,speakbool
    confirmstate = "nan"
    confirmstatenum = 0
    dataset_path = './people'
    dataset_path = './test'
    faces, labels = prepare_dataset(dataset_path)
    testfaces, testlabels = prepare_dataset(dataset_path)
    lbph, eigen = train_models(faces, labels)
    save_models(lbph, eigen)
    lbph, eigen = load_models()


    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


    counting = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        t1 = cv2.getTickCount()
        frame1 = frame

        if(counting<=3):
            try:
                lbph_accuracy, lbph_precision, lbph_recall, lbph_f1,eigen_accuracy,eigen_precision,eigen_recall,eigen_f1 = evaluate_models(lbph, eigen, testfaces, testlabels)
            except:
                print("evaluate_models issue ")
            counting = counting + 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            face_gray = gray[y:y+h, x:x+w]
            face_gray_resized = cv2.resize(face_gray, (100, 100))

            _1, lbph_pred = lbph.predict(face_gray_resized)
            _, eigen_pred = eigen.predict(face_gray_resized)
            lbph_label = datalabels[_1]
            eigen_label = datalabels[_]

            print(lbph_label, eigen_label) #120 3500
            label_text = ""
            lbph_label = lbph_label.split("_")[0]
            eigen_label=eigen_label.split("_")[0]

            label_text = eigen_label


            # if(lbph_pred<=120):
            #     label_text = lbph_label
            #     cv2.putText(frame, f'{lbph_label}', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            # elif(eigen_pred<=3500):
            #     label_text = eigen_label
            cv2.putText(frame, f'{label_text}', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            # cv2.putText(frame, f'labels : {_1}, {_}', (x, y-60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            # lbph_accuracy, lbph_precision, lbph_recall, lbph_f1,eigen_accuracy,eigen_precision,eigen_recall,eigen_f1

            # label_text = label_text.split("-")[0]
            print(label_text)

            if (confirmstatenum >= 10):
                        confirmstate = label_text
                        print(confirmstate)
                        if(speakbool == False):
                            speaktext = "Identified a face in front of you. It's "+str(label_text if (len(label_text)>0) else "unknown")+"."
                            speakbool = True
                        # gui.callGUI(label_text)
                        # while True:
                        #     if (gui.windowopenstate == 1):
                        #         print("Waiting for person vote")
                        #     else:
                        #         break
                        # confirmstatenum = 0
            else:
                print('Security check......')
                draw_label(frame, (x, y), label_text)
                confirmstatenum = confirmstatenum + 1


        cv2.putText(frame, f'distance : {distancedata:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        if(speakbool == False and distancedata<15):
            speaktext = "Obstacle a head with "+str(distancedata)+" distance"
            speakbool = True
        # cv2.putText(frame, f'lbph Accuracy: {lbph_accuracy:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        # cv2.putText(frame, f'lbph precision: {lbph_precision:.2f}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        # cv2.putText(frame, f'lbph recall: {lbph_recall:.2f}', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        # cv2.putText(frame, f'lbph F1 Score: {lbph_f1:.2f}', (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # cv2.putText(frame, f'eigen Accuracy: {eigen_accuracy:.2f}', (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        # cv2.putText(frame, f'eigen precision: {eigen_precision:.2f}', (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        # cv2.putText(frame, f'eigen recall: {eigen_recall:.2f}', (10, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        # cv2.putText(frame, f'eigen F1 Score: {eigen_f1:.2f}', (10, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
       
        # Acquire frame and resize to expected shape [1xHxWx3]
        frame = frame1.copy()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (width, height))
        input_data = np.expand_dims(frame_resized, axis=0)

        # Normalize pixel values if using a floating model (i.e. if model is non-quantized)
        if floating_model:
            input_data = (np.float32(input_data) - input_mean) / input_std

        # Perform the actual detection by running the model with the image as input
        interpreter.set_tensor(input_details[0]['index'],input_data)
        interpreter.invoke()

        # Retrieve detection results
        boxes = interpreter.get_tensor(output_details[boxes_idx]['index'])[0] # Bounding box coordinates of detected objects
        classes = interpreter.get_tensor(output_details[classes_idx]['index'])[0] # Class index of detected objects
        scores = interpreter.get_tensor(output_details[scores_idx]['index'])[0] # Confidence of detected objects

        # Loop over all detections and draw detection box if confidence is above minimum threshold
        for i in range(len(scores)):
            if ((scores[i] > min_conf_threshold) and (scores[i] <= 1.0)):

                # Get bounding box coordinates and draw box
                # Interpreter can return coordinates that are outside of image dimensions, need to force them to be within image using max() and min()
                ymin = int(max(1,(boxes[i][0] * imH)))
                xmin = int(max(1,(boxes[i][1] * imW)))
                ymax = int(min(imH,(boxes[i][2] * imH)))
                xmax = int(min(imW,(boxes[i][3] * imW)))
                
                cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), (10, 255, 0), 2)

                # Draw label
                object_name = labelsdect[int(classes[i])] # Look up object name from "labels" array using class index
                label = '%s: %d%%' % (object_name, int(scores[i]*100)) # Example: 'person: 72%'
                labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2) # Get font size
                label_ymin = max(ymin, labelSize[1] + 10) # Make sure not to draw label too close to top of window
                cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED) # Draw white box to put label text in
                cv2.putText(frame, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2) # Draw label text
                if(speakbool == False):
                    speaktext = "Detected a "+label+" in your front"
                    speakbool = True
        # Draw framerate in corner of frame
        cv2.putText(frame,'FPS: {0:.2f}'.format(frame_rate_calc),(30,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),2,cv2.LINE_AA)

        # All the results have been drawn on the frame, so it's time to display it.
        cv2.imshow('Object detector', frame)

        # Calculate framerate
        t2 = cv2.getTickCount()
        time1 = (t2-t1)/freq
        frame_rate_calc= 1/time1

        cv2.imshow('Video', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    thread1 = threading.Thread(target=measure)
    thread2 = threading.Thread(target=main)
    sentiment_thread = threading.Thread(target=speak)
    thread1.daemon = True  # Make the thread a daemon so that it exits when the main program exits
    thread2.daemon = True
    sentiment_thread.daemon = True

    sentiment_thread.start()

    thread1.start()
    thread2.start()

    thread1.join()
    thread2.join()
    sentiment_thread.join()
