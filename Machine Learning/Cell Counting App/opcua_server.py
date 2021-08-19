from opcua import Server
import RPi.GPIO as GPIO
import time

server = Server()
url = "opc.tcp://<ip here>"
server.set_endpoint(url)

name = "OPCUA_SERVER"
addspace = server.register_namespace(name)

node = server.get_objects_node()

ServerInfo = node.add_object(addspace, "OPCUA Simulation Server")
param = node.add_object(addspace, "parameters")

donebit = param.add_variable(addspace, "LED_STATUS", 0)
donebit.set_writable()
startbit = param.add_variable(addspace, "START_BUTTON_STATUS", 0)
startbit.set_writable()

server.start()
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)
LED = 17
BUTTON = 21
GPIO.setup(LED, GPIO.OUT)
GPIO.setup(BUTTON, GPIO.IN, pull_up_down=GPIO.PUD_UP)

try:
    while True:
        gled = donebit.get_value()
    
        if gled == 1:
            GPIO.output(LED,GPIO.HIGH)
        elif gled == 0:
            GPIO.output(LED,GPIO.LOW)
        else:
            GPIO.output(LED,GPIO.LOW)
        
        
        if GPIO.input(BUTTON) == 1:
            startbit.set_value(0)
            time.sleep(1)
        
        elif GPIO.input(BUTTON) == 0:
            startbit.set_value(1)
            time.sleep(1)
        
        time.sleep(1)
        
except:
    print(" ") 

finally:
    
    GPIO.cleanup()


