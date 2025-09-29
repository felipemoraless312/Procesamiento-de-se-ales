import numpy as np
import matplotlib.pyplot as plt

# Parámetros de la señal original
A1, f1 = 1.0, 1200.0   # Hz
A2, f2 = 0.6, 3500.0   # Hz

# Parámetros de muestreo
fs = 2000.0            # prueba con 2000 (aliasing) o 3000 (sin aliasing para 1200)
T = 0.05               # duración en segundos (suficiente para buena resolución)
t_cont = np.arange(0, T, 1/(10*fs))  # tiempo "continuo" para referencia (oversample)
x_cont = A1*np.cos(2*np.pi*f1*t_cont) + A2*np.cos(2*np.pi*f2*t_cont)

# Secuencia muestreada
t = np.arange(0, T, 1/fs)
x = A1*np.cos(2*np.pi*f1*t) + A2*np.cos(2*np.pi*f2*t)

# DFT (FFT)
N = 2**14
X = np.fft.rfft(x * np.hanning(len(x)), n=N)
freqs = np.fft.rfftfreq(N, d=1/fs)
mag = np.abs(X) / np.max(np.abs(X))

# Plots
plt.figure(figsize=(12,6))

plt.subplot(2,1,1)
plt.plot(t_cont, x_cont, label='x(t) continua (oversampled)')
plt.plot(t, x, 'o-', label=f'muestras fs={fs} Hz', markersize=3)
plt.xlim(0, 0.01)
plt.legend()
plt.title('Señal continua y muestras')

plt.subplot(2,1,2)
plt.plot(freqs, mag)
plt.xlim(0, fs/2)
plt.xlabel('Frecuencia (Hz)')
plt.ylabel('Magnitud normalizada')
plt.title('Espectro de la señal muestreada')
plt.grid(True)
# marcar frecuencias esperadas
plt.axvline(800, color='k', linestyle='--', linewidth=0.8)   # primer caso (fs=2000 -> 1200->800)
plt.axvline(500, color='r', linestyle='--', linewidth=0.8)   # 3500 -> 500 (fs=2000)
plt.show()
