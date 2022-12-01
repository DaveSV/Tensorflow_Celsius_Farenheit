import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import tensorflow as tf
import numpy as np

celsius = np.array([-40, -10, 0, 8, 15, 22, 38], dtype=float)
fahrenheit = np.array([-40, 14, 32, 46, 59, 72, 100], dtype=float)


red1 = tf.keras.layers.Dense(units=3, input_shape=[1])
red2 = tf.keras.layers.Dense(units=3)
salida = tf.keras.layers.Dense(units=1)
modelo = tf.keras.Sequential([red1, red2, salida])

modelo.compile(
    optimizer=tf.keras.optimizers.Adam(0.1),
    loss='mean_squared_error'
)

print("Comenzando entrenamiento...")
historial = modelo.fit(celsius, fahrenheit, epochs=1000, verbose=False)
print("Modelo entrenado! Mostrando gráfico de aprendizaje...")


import matplotlib.pyplot as plt
plt.xlabel("# Medición")
plt.ylabel("Magnitud de pérdida")
plt.plot(historial.history["loss"])
plt.show()

print("Comenzando predicción!")
grados = int(input('Ingrese grados celsius:'))
resultado = modelo.predict([grados])
print("El resultado es " + str(np.ndarray.round(resultado,0)).replace(' [', '').replace('[', '').replace(']', '').replace('.', '') + " fahrenheit!")

