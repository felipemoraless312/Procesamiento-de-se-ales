"""
    @author: Luis Felipe Morales Gutierrez
    
        15-10-25.py: Generar y graficar señales seno y cuadrada, y simular señal ECG con ruido y filtrado.

    description: Este script genera señales seno, cuadrada y coseno, las grafica,
    y simula una señal ECG con ruido, aplicando un filtro pasa bajos.
"""
import numpy as np
import matplotlib.pyplot as plt

#Definiendo constantes

# Parámetros de la señal 
fs = 2000  # Frecuencia de muestreo en Hz
duracion = 1  # Duración en segundos
frecuencia_seno = 60  # Frecuencia de la señal seno en Hz
frecuencia_cuadrada = 60  # Frecuencia de la señal cuadrada en Hz
frecuencia_coseno = 90  # Frecuencia de la señal coseno en Hz
amplitud = 0.3  # Amplitud de las señales
t = np.arange(0, duracion, 1/fs)  # Vector de tiempo

# Generar señales
senal_seno = amplitud * np.sin(2 * np.pi * frecuencia_seno * t)
senal_cuadrada = amplitud * np.sign(np.sin(2 * np.pi * frecuencia_cuadrada * t))
senal_coseno = amplitud * np.cos(2 * np.pi * frecuencia_coseno * t)

# Graficar señales
"""
plt.figure(figsize=(12, 8))
plt.subplot(3, 1, 1)
plt.plot(t, senal_seno, label='Señal Seno', color='blue')
plt.title('Señal Seno')
plt.xlabel('Tiempo (s)')
plt.ylabel('Amplitud')
plt.grid()
plt.subplot(3, 1, 2)
plt.plot(t, senal_cuadrada, label='Señal Cuadrada', color='orange')
plt.title('Señal Cuadrada')
plt.xlabel('Tiempo (s)')
plt.ylabel('Amplitud')
plt.grid()
plt.subplot(3, 1, 3)
plt.plot(t, senal_coseno, label='Señal Coseno', color='green')
plt.title('Señal Coseno')
plt.xlabel('Tiempo (s)')
plt.ylabel('Amplitud')
plt.grid()
plt.tight_layout()
plt.show()
"""
# Señal compuesta
"""una señal compuesta es la suma de varias señales individuales"""
signal = senal_seno
"""
#grafica de senal superpuesta

plt.figure(figsize=(12, 4))
plt.plot(t, signal, label='Señal Compuesta', color='purple')
plt.show()
"""
#FFT
"""

frequencies = np.fft.rfftfreq(len(signal), 1/fs)#frecuencias positivas
fft_values = np.fft.rfft(signal) #valores de la FFT
magnitude = np.abs(fft_values)#magnitud de la FFT
#graficar FFT
plt.figure(figsize=(12, 4))
plt.plot(frequencies, magnitude, color='red')
plt.show()"""

#simular ecg con ruido
#"""

import neurokit2 as nk

ecg = nk.ecg_simulate(duration=duracion, sampling_rate=fs, heart_rate=120)

ecg_noise = ecg + 0.1 * np.random.normal(size=len(ecg))
#ecg_noise = ecg + signal


plt.figure(figsize=(12, 4))
plt.plot(t, ecg_noise, label='ECG con Ruido', color='gray')
plt.show()


#fft
frequencies = np.fft.rfftfreq(len(ecg_noise), 1/fs)#frecuencias positivas
fft_values = np.fft.rfft(ecg_noise) #valores de la FFT
magnitude = np.abs(fft_values)#magnitud de la FFT
#graficar FFT
plt.figure(figsize=(12, 4))
plt.plot(frequencies, magnitude, color='red')
plt.show()

#filtro pasa bajos
ecg_filtered = nk.signal_filter(ecg_noise, sampling_rate=fs, lowcut=0.5, highcut=20, method='butterworth', order=2)
plt.figure(figsize=(12, 4))
plt.plot(t, ecg_filtered, label='ECG Filtrado', color='blue')
plt.show()
#"""
