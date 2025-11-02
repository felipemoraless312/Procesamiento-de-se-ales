import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog

"""
     1. Abrir ventana para seleccionar archivo ---
        Crea una ventana de tkinter"""
root = tk.Tk()
root.withdraw()  # Oculta la ventana principal
file_path = filedialog.askopenfilename(
    title="Selecciona el archivo de EMG",
    filetypes=[("Archivos de texto o CSV", "*.txt *.csv"), ("Todos los archivos", "*.*")]
)

if not file_path:
    print("No se seleccionó ningún archivo.")
    exit()

print(f"Archivo seleccionado: {file_path}")

# --- 2. Cargar los datos ---
try:
    # Intenta cargar el archivo como CSV o TXT
    data = np.loadtxt(file_path, delimiter=',')
except ValueError:
    # Si no hay comas, intenta con espacios
    data = np.loadtxt(file_path)

# --- 3. Procesar los datos ---
# Si el archivo tiene varias columnas, usa solo la primera
if data.ndim > 1:
    emg_signal = data[:, 0]
else:
    emg_signal = data

# Parámetros de muestreo (ajústalo )
fs = 1000  # frecuencia de muestreo
t = np.arange(len(emg_signal)) / fs

# --- 4. Graficar ---
plt.figure(figsize=(10, 4))
plt.plot(t, emg_signal, color='purple')
plt.title("Señal EMG")
plt.xlabel("Tiempo (s)")
plt.ylabel("Amplitud")
plt.grid(True)
plt.tight_layout()
plt.show()

# --- 5. Análisis básico ---
mean_emg = np.mean(emg_signal)
std_emg = np.std(emg_signal)
print(f"Media de la señal EMG: {mean_emg:.4f}")
print(f"Desviación estándar de la señal EMG: {std_emg:.4f}")
max_emg = np.max(emg_signal)
min_emg = np.min(emg_signal)
print(f"Valor máximo de la señal EMG: {max_emg:.4f}")
print(f"Valor mínimo de la señal EMG: {min_emg:.4f}")
rms_emg = np.sqrt(np.mean(emg_signal**2))
print(f"Valor RMS de la señal EMG: {rms_emg:.4f}")
