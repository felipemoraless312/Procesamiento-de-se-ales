import numpy as np
import neurokit2 as nk
import matplotlib.pyplot as plt

# Parámetros de la señal ecg
fs= 1000
duration = 2
frec_ecg = 90
t = np.arange(0, duration, 1/fs)

#señal seno 50 hz
frec_seno= 50
a_seno= 0.5

#seno 100 hz
seno_100= 100
a_seno100= 0.3

#ruido gaussiano
#nombre-variable= amplitud * np.random.normal(size=len(t))
ruido_gaussiano = 0.2 * np.random.normal(size=len(t))

#senales de senoides
senal_sen50 = a_seno * np.sin(2 * np.pi * frec_seno * t)
senal_sen100 = a_seno100 * np.sin(2 * np.pi * seno_100 * t) 

#senal ecg
#parametros: duration, sampling_rate, heart_rate
ecg = nk.ecg_simulate(duration=duration, sampling_rate=fs, heart_rate=frec_ecg)

compuesta = senal_sen50 + senal_sen100 + ruido_gaussiano + ecg

#fft de señal compuesta
fft_compuesta = np.fft.fft(compuesta)
frequencies = np.fft.fftfreq(len(fft_compuesta), 1/fs)
magnitude = np.abs(fft_compuesta)

#filtro de la señal ecg compuesta
#parametros: signal, sampling_rate, lowcut, highcut, method, order
ecg_filtered = nk.signal_filter(compuesta, sampling_rate=fs, lowcut=0.5, highcut=20, method='butterworth', order=2)
#graficas
"""
plt.figure(figsize=(14, 10))
plt.subplot(4, 1, 1)
plt.plot(t, ecg, color='green', linewidth=0.8)
plt.title('ECG Original Limpio', fontsize=12, fontweight='bold')
plt.ylabel('Amplitud')

plt.grid(True, alpha=0.3)
plt.xlim([0, 1.5]) # Mostrar solo 3 segundos para mejor visualización
plt.subplot(4, 1, 2)
plt.plot(t, compuesta, color='red', linewidth=0.8)
plt.title('ECG Contaminado', fontsize=12, fontweight='bold')
plt.ylabel('Amplitud')
plt.grid(True, alpha=0.3)
plt.xlim([0, 3])
plt.subplot(4, 1, 3)
plt.plot(frequenc   ies, magnitude, color='orange', linewidth=0.8)
plt.title('Espectro FFT de ECG Contaminado', fontsize=12, fontweight='bold')
plt.xlabel('Frecuencia (Hz)')
plt.ylabel('Magnitud')
plt.grid(True, alpha=0.3)
plt.xlim([0, 150]) # Mostrar hasta 150 Hz

plt.subplot(4, 1, 4)
plt.plot(t, ecg_filtered, color='blue', linewidth=0.8)
plt.title('ECG Filtrado', fontsize=12, fontweight='bold')
plt.xlabel('Tiempo (s)')
plt.ylabel('Amplitud')
plt.grid(True, alpha=0.3)
plt.xlim([0, 3])
plt.tight_layout()
plt.show()
"""
plt.figure(figsize=(12, 8))
plt.plot(t, ecg, label='ECG Original Limpio', color='orange')
plt.xlim(0,1.5)
plt.show()
