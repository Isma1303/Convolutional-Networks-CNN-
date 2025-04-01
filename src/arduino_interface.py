import serial

def connect_arduino(port="COM3", baudrate=9600):
    try:
        arduino = serial.Serial(port, baudrate, timeout=1)
        print("Conectado a Arduino")
        return arduino
    except Exception as e:
        print(f"Error: {e}")
        return None

arduino = connect_arduino()
if arduino:
    arduino.write(b"Iniciando detección")
