import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# --- Configuración de la señal ---
f0 = 5                # frecuencia de la señal (Hz)
t_cont = np.linspace(0, 1, 1000)  # tiempo continuo
signal_cont = np.sin(2 * np.pi * f0 * t_cont)  # señal continua

# --- Función para muestrear y reconstruir ---
def muestrear_y_reconstruir(fs):
    t_sample = np.arange(0, 1, 1/fs)
    signal_sample = np.sin(2 * np.pi * f0 * t_sample)
    
    # Interpolación lineal para reconstruir señal
    f_interp = interp1d(t_sample, signal_sample, kind='linear', fill_value="extrapolate")
    signal_recon = f_interp(t_cont)
    
    # Error de reconstrucción
    error = np.sqrt(np.mean((signal_cont - signal_recon)**2))
    
    return t_sample, signal_sample, signal_recon, error

# --- Frecuencias de muestreo para probar ---
fs_values = [12, 10, 8, 6]  # fs > 2*f0, fs ~ 2*f0, fs < 2*f0
errors = []

plt.figure(figsize=(12, 10))

for i, fs in enumerate(fs_values, 1):
    t_s, s_s, s_r, err = muestrear_y_reconstruir(fs)
    errors.append(err)
    
    plt.subplot(len(fs_values), 1, i)
    plt.plot(t_cont, signal_cont, label='Señal continua', color='blue')
    plt.stem(t_s, s_s, linefmt='r-', markerfmt='ro', basefmt=" ", label='Muestras')
    plt.plot(t_cont, s_r, '--', color='green', label='Reconstrucción')
    plt.title(f'fs = {fs} Hz, Error RMS = {err:.3f}')
    plt.legend()
    plt.grid(True)

plt.tight_layout()
plt.show()