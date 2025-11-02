import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import neurokit2 as nk

# Configuración
fs = 1000  # Frecuencia de muestreo (Hz)
duracion = 10  # Duración en segundos

# 1. Generar señal ECG sintética
ecg_limpio = nk.ecg_simulate(duration=duracion, sampling_rate=fs, heart_rate=70)

# 2. Añadir diferentes tipos de ruido
amplitud_ruido = 0.3
ruido_blanco = np.random.normal(0, amplitud_ruido, len(ecg_limpio))

# Ruido de baja frecuencia (simula movimiento de línea base)
t = np.arange(len(ecg_limpio)) / fs
ruido_lf = 0.3 * np.sin(2 * np.pi * 0.5 * t)

# Ruido de alta frecuencia (simula interferencia muscular)
ruido_hf = 0.05 * np.sin(2 * np.pi * 60 * t)

# Señal con ruido
ecg_ruidoso = ecg_limpio + ruido_blanco + ruido_lf + ruido_hf
# 3. Calcular la Transformada de Fourier

fft_limpio = np.fft.fft(ecg_limpio)
fft_ruidoso = np.fft.fft(ecg_ruidoso)
frecuencias = np.fft.fftfreq(len(ecg_limpio), 1/fs)

# Tomar solo la mitad positiva del espectro
mitad = len(frecuencias) // 2
frecuencias = frecuencias[:mitad]
fft_limpio = np.abs(fft_limpio[:mitad])
fft_ruidoso = np.abs(fft_ruidoso[:mitad])

# 4. Filtrar la señal ruidosa (filtro pasa banda)
# Frecuencias típicas del ECG: 0.5 - 40 Hz
sos = signal.butter(4, [0.5, 40], btype='bandpass', fs=fs, output='sos')
ecg_filtrado = signal.sosfilt(sos, ecg_ruidoso)

fft_filtrado = np.abs(np.fft.fft(ecg_filtrado)[:mitad])

# 5. Visualización de resultados
fig, axes = plt.subplots(3, 2, figsize=(15, 12))
fig.suptitle('Análisis de Señal ECG y Transformada de Fourier', fontsize=16, fontweight='bold')

# Tiempo para graficar (primeros 3 segundos)
tiempo_plot = 3
muestras_plot = int(tiempo_plot * fs)
tiempo = np.arange(muestras_plot) / fs

# Fila 1: ECG limpio
axes[0, 0].plot(tiempo, ecg_limpio[:muestras_plot], 'b', linewidth=1.5)
axes[0, 0].set_title('ECG Limpio (Dominio del Tiempo)')
axes[0, 0].set_xlabel('Tiempo (s)')
axes[0, 0].set_ylabel('Amplitud')
axes[0, 0].grid(True, alpha=0.3)

axes[0, 1].plot(frecuencias, fft_limpio, 'b', linewidth=1.5)
axes[0, 1].set_title('FFT del ECG Limpio')
axes[0, 1].set_xlabel('Frecuencia (Hz)')
axes[0, 1].set_ylabel('Magnitud')
axes[0, 1].set_xlim([0, 100])
axes[0, 1].grid(True, alpha=0.3)

# Fila 2: ECG con ruido
axes[1, 0].plot(tiempo, ecg_ruidoso[:muestras_plot], 'r', linewidth=1.5)
axes[1, 0].set_title('ECG con Ruido (Dominio del Tiempo)')
axes[1, 0].set_xlabel('Tiempo (s)')
axes[1, 0].set_ylabel('Amplitud')
axes[1, 0].grid(True, alpha=0.3)

axes[1, 1].plot(frecuencias, fft_ruidoso, 'r', linewidth=1.5)
axes[1, 1].set_title('FFT del ECG con Ruido')
axes[1, 1].set_xlabel('Frecuencia (Hz)')
axes[1, 1].set_ylabel('Magnitud')
axes[1, 1].set_xlim([0, 100])
axes[1, 1].grid(True, alpha=0.3)

# Fila 3: ECG filtrado
axes[2, 0].plot(tiempo, ecg_filtrado[:muestras_plot], 'g', linewidth=1.5)
axes[2, 0].set_title('ECG Filtrado (Dominio del Tiempo)')
axes[2, 0].set_xlabel('Tiempo (s)')
axes[2, 0].set_ylabel('Amplitud')
axes[2, 0].grid(True, alpha=0.3)

axes[2, 1].plot(frecuencias, fft_filtrado, 'g', linewidth=1.5)
axes[2, 1].set_title('FFT del ECG Filtrado')
axes[2, 1].set_xlabel('Frecuencia (Hz)')
axes[2, 1].set_ylabel('Magnitud')
axes[2, 1].set_xlim([0, 100])
axes[2, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# 6. Análisis estadístico
print("="*60)
print("ANÁLISIS DE LA SEÑAL ECG")
print("="*60)
print(f"\nDuración: {duracion} segundos")
print(f"Frecuencia de muestreo: {fs} Hz")
print(f"Total de muestras: {len(ecg_limpio)}")
print(f"\nSNR (Signal-to-Noise Ratio):")
print(f"  - Señal con ruido: {10 * np.log10(np.var(ecg_limpio) / np.var(ruido_blanco + ruido_lf + ruido_hf)):.2f} dB")
print(f"\nFrecuencia cardíaca simulada: 70 BPM")
print(f"Frecuencias dominantes esperadas: ~1.17 Hz (frecuencia cardíaca) y sus armónicos")
print("="*60)
