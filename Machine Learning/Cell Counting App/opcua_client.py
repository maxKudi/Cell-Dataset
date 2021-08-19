from opcua import Client
import time

url = "opc.tcp://<ip here>"
    client = Client(url)
    client.connect()
    mydonebit = client.get_node("ns=2; i=3")
    mystartbit = client.get_node("ns=2; i=4")
    mystartbit = bool(mystartbit.get_data_value())

while True:
	myled = client.get_node("ns=2; i=3")
	led = myled.set_value(1)

	time.sleep(1)