import mysql.connector

def get_connection():
    try:
        conn = mysql.connector.connect(
            host="localhost",
            user="root",      
            password="", 
            database="cnn"
        )
        if conn.is_connected():
            print("Conexión exitosa a la base de datos")
            return conn
        else:
            print("Error al conectar a la base de datos")
            conn.close()  
            return None
    except mysql.connector.Error as err:
        print(f"Error de conexión: {err}")
        return None