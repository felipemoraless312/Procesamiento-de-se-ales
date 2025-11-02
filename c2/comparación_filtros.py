import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import neurokit2 as nk

# Configuración
fs = 1000  # Frecuencia de muestreo (Hz)
duracion = 10  # Duración en segundos

# 1. Generar señal ECG sintética limpia
ecg_limpio = nk.ecg_simulate(duration=duracion, sampling_rate=fs, heart_rate=70)

# 2. Aplicar 4 filtros diferentes al ECG limpio

# Filtro 1: Pasa banda Butterworth (0.5 - 40 Hz)
sos1 = signal.butter(4, [0.5, 40], btype='bandpass', fs=fs, output='sos')
ecg_filtro1 = signal.sosfilt(sos1, ecg_limpio)

# Filtro 2: Pasa banda Chebyshev Tipo I (1 - 35 Hz)
sos2 = signal.cheby1(4, 0.1, [1, 35], btype='bandpass', fs=fs, output='sos')
ecg_filtro2 = signal.sosfilt(sos2, ecg_limpio)

# Filtro 3: Pasa alto + Pasa bajo (Cascada de dos filtros)
# Pasa alto: 0.5 Hz
sos3_hp = signal.butter(3, 0.5, btype='high', fs=fs, output='sos')
# Pasa bajo: 40 Hz
sos3_lp = signal.butter(3, 40, btype='low', fs=fs, output='sos')
ecg_filtro3 = signal.sosfilt(sos3_hp, ecg_limpio)
ecg_filtro3 = signal.sosfilt(sos3_lp, ecg_filtro3)

# Filtro 4: Pasa banda Elíptico (0.5 - 40 Hz)
sos4 = signal.ellip(4, 0.1, 40, [0.5, 40], btype='bandpass', fs=fs, output='sos')
ecg_filtro4 = signal.sosfilt(sos4, ecg_limpio)

# 3. Calcular FFT de todas las señales
fft_limpio = np.abs(np.fft.fft(ecg_limpio))
fft_f1 = np.abs(np.fft.fft(ecg_filtro1))
fft_f2 = np.abs(np.fft.fft(ecg_filtro2))
fft_f3 = np.abs(np.fft.fft(ecg_filtro3))
fft_f4 = np.abs(np.fft.fft(ecg_filtro4))

frecuencias = np.fft.fftfreq(len(ecg_limpio), 1/fs)

# Tomar solo la mitad positiva del espectro
mitad = len(frecuencias) // 2
frecuencias = frecuencias[:mitad]
fft_limpio = fft_limpio[:mitad]
fft_f1 = fft_f1[:mitad]
fft_f2 = fft_f2[:mitad]
fft_f3 = fft_f3[:mitad]
fft_f4 = fft_f4[:mitad]

# 4. Visualización de resultados
fig, axes = plt.subplots(5, 2, figsize=(15, 16))
fig.suptitle('Análisis de Señal ECG Limpia con 4 Filtros Diferentes', fontsize=16, fontweight='bold')

# Tiempo para graficar (primeros 3 segundos)
tiempo_plot = 3
muestras_plot = int(tiempo_plot * fs)
tiempo = np.arange(muestras_plot) / fs

# Fila 0: ECG limpio original
axes[0, 0].plot(tiempo, ecg_limpio[:muestras_plot], 'b', linewidth=1.5)
axes[0, 0].set_title('ECG Original (Dominio del Tiempo)')
axes[0, 0].set_xlabel('Tiempo (s)')
axes[0, 0].set_ylabel('Amplitud')
axes[0, 0].grid(True, alpha=0.3)

axes[0, 1].plot(frecuencias, fft_limpio, 'b', linewidth=1.5)
axes[0, 1].set_title('FFT del ECG Original')
axes[0, 1].set_xlabel('Frecuencia (Hz)')
axes[0, 1].set_ylabel('Magnitud')
axes[0, 1].set_xlim([0, 100])
axes[0, 1].grid(True, alpha=0.3)

# Fila 1: Filtro Butterworth pasa banda
axes[1, 0].plot(tiempo, ecg_filtro1[:muestras_plot], 'g', linewidth=1.5)
axes[1, 0].set_title('Filtro 1: Butterworth Pasa Banda (0.5-40 Hz)')
axes[1, 0].set_xlabel('Tiempo (s)')
axes[1, 0].set_ylabel('Amplitud')
axes[1, 0].grid(True, alpha=0.3)

axes[1, 1].plot(frecuencias, fft_f1, 'g', linewidth=1.5)
axes[1, 1].set_title('FFT - Filtro Butterworth')
axes[1, 1].set_xlabel('Frecuencia (Hz)')
axes[1, 1].set_ylabel('Magnitud')
axes[1, 1].set_xlim([0, 100])
axes[1, 1].grid(True, alpha=0.3)

# Fila 2: Filtro Chebyshev pasa banda
axes[2, 0].plot(tiempo, ecg_filtro2[:muestras_plot], 'r', linewidth=1.5)
axes[2, 0].set_title('Filtro 2: Chebyshev Pasa Banda (1-35 Hz)')
axes[2, 0].set_xlabel('Tiempo (s)')
axes[2, 0].set_ylabel('Amplitud')
axes[2, 0].grid(True, alpha=0.3)

axes[2, 1].plot(frecuencias, fft_f2, 'r', linewidth=1.5)
axes[2, 1].set_title('FFT - Filtro Chebyshev')
axes[2, 1].set_xlabel('Frecuencia (Hz)')
axes[2, 1].set_ylabel('Magnitud')
axes[2, 1].set_xlim([0, 100])
axes[2, 1].grid(True, alpha=0.3)

# Fila 3: Filtro Cascada (Alto + Bajo)
axes[3, 0].plot(tiempo, ecg_filtro3[:muestras_plot], 'orange', linewidth=1.5)
axes[3, 0].set_title('Filtro 3: Cascada Alto (0.5 Hz) + Bajo (40 Hz)')
axes[3, 0].set_xlabel('Tiempo (s)')
axes[3, 0].set_ylabel('Amplitud')
axes[3, 0].grid(True, alpha=0.3)

axes[3, 1].plot(frecuencias, fft_f3, 'orange', linewidth=1.5)
axes[3, 1].set_title('FFT - Filtro Cascada')
axes[3, 1].set_xlabel('Frecuencia (Hz)')
axes[3, 1].set_ylabel('Magnitud')
axes[3, 1].set_xlim([0, 100])
axes[3, 1].grid(True, alpha=0.3)

# Fila 4: Filtro Elíptico pasa banda
axes[4, 0].plot(tiempo, ecg_filtro4[:muestras_plot], 'purple', linewidth=1.5)
axes[4, 0].set_title('Filtro 4: Elíptico Pasa Banda (0.5-40 Hz)')
axes[4, 0].set_xlabel('Tiempo (s)')
axes[4, 0].set_ylabel('Amplitud')
axes[4, 0].grid(True, alpha=0.3)

axes[4, 1].plot(frecuencias, fft_f4, 'purple', linewidth=1.5)
axes[4, 1].set_title('FFT - Filtro Elíptico')
axes[4, 1].set_xlabel('Frecuencia (Hz)')
axes[4, 1].set_ylabel('Magnitud')
axes[4, 1].set_xlim([0, 100])
axes[4, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# 5. Análisis estadístico y comparación
print("="*70)
print("ANÁLISIS COMPARATIVO DE FILTROS APLICADOS AL ECG LIMPIO")
print("="*70)
print(f"\nDuración: {duracion} segundos")
print(f"Frecuencia de muestreo: {fs} Hz")
print(f"Total de muestras: {len(ecg_limpio)}")
print(f"Frecuencia cardíaca simulada: 70 BPM (~1.17 Hz)")

# Calcular métricas para cada filtro
def calcular_metricas(señal_original, señal_filtrada, nombre_filtro):
    error_rmse = np.sqrt(np.mean((señal_original - señal_filtrada)**2))
    correlacion = np.corrcoef(señal_original, señal_filtrada)[0, 1]
    print(f"\n{nombre_filtro}:")
    print(f"  - Error RMSE: {error_rmse:.6f}")
    print(f"  - Correlación: {correlacion:.6f}")
    print(f"  - Amplitud máxima: {np.max(np.abs(señal_filtrada)):.6f}")

calcular_metricas(ecg_limpio, ecg_filtro1, "Filtro 1: Butterworth")
calcular_metricas(ecg_limpio, ecg_filtro2, "Filtro 2: Chebyshev")
calcular_metricas(ecg_limpio, ecg_filtro3, "Filtro 3: Cascada (Alto+Bajo)")
calcular_metricas(ecg_limpio, ecg_filtro4, "Filtro 4: Elíptico")

print("\n" + "="*70)
