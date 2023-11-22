
import RPi.GPIO as GPIO 

import time 
from subprocess import call

GPIO.setwarnings(False) 
GPIO.setmode(GPIO.BCM) 

TRIG = 18  # Trigger pin of the Ultrasonic Sensor 
ECHO = 23 #Echo pin of the Ultrasonic Sensor 
GPIO.setup(TRIG,GPIO.OUT) 
GPIO.setup(ECHO,GPIO.IN) 

#Connects to the vehicle using serial po
    
def measure(): 
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
        #Sends measured distance(dist1) and orientation(0) as a mavlink message. 
    else:
        dist1 = 0
    return dist1 # Main code 


while True: 
    print("started")
    d=measure() 
    time.sleep(0.1) 
    call(['espeak " distance equal to '+str(d)+'" 2>/dev/null'], shell=True)
    print(d)