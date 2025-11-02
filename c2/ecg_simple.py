import neurokit2 as nk
import matplotlib.pyplot as plt

import numpy as np
# Simulate ECG signal
fs=1000 #frecuencia de muestreo
duracion=10 #duración en segundos
frecuencia_cardiaca=120 #frecuencia cardiaca en latidos por minuto
#generar señan sinoidal

#señal seno y cuadrada

ecg = nk.ecg_simulate(duration=duracion, sampling_rate=fs, heart_rate=frecuencia_cardiaca)
print(f"señal generada: {len(ecg)} muestras {duracion} segundos")

#crear array de tiempo
tiempo = np.arange(len(ecg)) / fs
print(f"tiempo generado: {len(tiempo)} muestras {duracion} segundos")

#graficar señal ecg
plt.figure(figsize=(12,4))
plt.plot(tiempo, ecg, linewidth=1.5, color='blue')
plt.xlabel('Tiempo (s)')#nombre de etiqueta del eje x
plt.ylabel('Amplitud (mv)')#nombre de etiqueta del eje y
plt.title('Señal ECG Sintética')#título del gráfico o figura
plt.grid(True, alpha=0.3)#mostrar cuadrícula
plt.tight_layout()#ajustar diseño
plt.show()#mostrar gráfico

#señales seno y cuadrada

#FFT
#filtros
