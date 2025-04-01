#define LED_VERDE 2  // LED verde para predicciones
#define LED_ROJO 12  // LED rojo si el programa no se ejecuta

unsigned long ultimoTiempo = 0; // Última vez que se recibió un dato
const unsigned long tiempoLimite = 5000; // 5 segundos sin datos

void setup() {
    Serial.begin(9600); 
    pinMode(LED_VERDE, OUTPUT);
    pinMode(LED_ROJO, OUTPUT);
    digitalWrite(LED_VERDE, LOW);
    digitalWrite(LED_ROJO, LOW);
}

void loop() {
    if (Serial.available() > 0) { 
        String data = Serial.readStringUntil('\n'); 
        Serial.print("Recibido: ");
        Serial.println(data);
        ultimoTiempo = millis(); // Actualizar tiempo

        // Si el objeto detectado es "perro", encender LED verde
        if (data == "perro") {
            digitalWrite(LED_VERDE, HIGH);
        } else {
            digitalWrite(LED_VERDE, LOW);
        }
    }

    // Si han pasado más de 5 segundos sin recibir datos, encender LED rojo
    if (millis() - ultimoTiempo > tiempoLimite) {
        digitalWrite(LED_ROJO, HIGH);  // Indicar error
    } else {
        digitalWrite(LED_ROJO, LOW);   // Apagar LED rojo si se restablece
    }
}
