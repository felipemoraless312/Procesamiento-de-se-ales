import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from matplotlib.animation import FuncAnimation
import matplotlib.gridspec as gridspec
import neurokit2 as nk

# Configuración inicial
fig = plt.figure(figsize=(14, 10))
fig.suptitle('Análisis FFT de Señal ECG con Ruido en Tiempo Real', fontsize=16, fontweight='bold')

# Crear grid para organizar los subplots
gs = gridspec.GridSpec(4, 2, figure=fig, hspace=0.4, wspace=0.3, 
                       height_ratios=[1, 1, 1, 1.5])

# Crear subplots
ax_ecg_clean = fig.add_subplot(gs[0, :])  # ECG limpia
ax_ecg_noisy = fig.add_subplot(gs[1, :])  # ECG con ruido
ax_fft = fig.add_subplot(gs[2, :])  # Transformada de Fourier

# Generar señal ECG sintética usando NeuroKit2
sampling_rate = 500
duration = 5  # segundos
ecg_signal = nk.ecg_simulate(duration=duration, sampling_rate=sampling_rate, heart_rate=70)

# Normalizar la señal ECG
ecg_signal = (ecg_signal - np.mean(ecg_signal)) / np.std(ecg_signal)

# Crear vector de tiempo
t = np.linspace(0, duration, len(ecg_signal))

# Parámetros iniciales de ruido
noise_50hz_init = 0.0  # Ruido de línea eléctrica (50 Hz)
noise_60hz_init = 0.0  # Ruido de línea eléctrica (60 Hz)
noise_baseline_init = 0.0  # Ruido de línea base (0.5 Hz)
noise_muscle_init = 0.0  # Ruido muscular (20-40 Hz)
noise_white_init = 0.0  # Ruido blanco

is_playing = False

# Función para generar ruidos
def generate_noise(amp_50hz, amp_60hz, amp_baseline, amp_muscle, amp_white):
    # Ruido de línea eléctrica 50 Hz
    noise_50 = amp_50hz * np.sin(2 * np.pi * 50 * t)
    
    # Ruido de línea eléctrica 60 Hz
    noise_60 = amp_60hz * np.sin(2 * np.pi * 60 * t)
    
    # Ruido de línea base (deriva lenta)
    noise_base = amp_baseline * np.sin(2 * np.pi * 0.5 * t)
    
    # Ruido muscular (mezcla de frecuencias 20-40 Hz)
    noise_musc = amp_muscle * (
        np.sin(2 * np.pi * 25 * t) + 
        0.5 * np.sin(2 * np.pi * 30 * t) +
        0.3 * np.sin(2 * np.pi * 35 * t)
    )
    
    # Ruido blanco (gaussiano)
    noise_wht = amp_white * np.random.randn(len(t))
    
    total_noise = noise_50 + noise_60 + noise_base + noise_musc + noise_wht
    return total_noise

# Función para calcular FFT
def calculate_fft(signal, sample_rate):
    n = len(signal)
    fft_values = np.fft.fft(signal)
    fft_freq = np.fft.fftfreq(n, 1/sample_rate)
    
    # Tomar solo frecuencias positivas
    positive_freq_idx = fft_freq > 0
    frequencies = fft_freq[positive_freq_idx]
    magnitudes = np.abs(fft_values[positive_freq_idx]) * 2 / n
    
    # Limitar a frecuencias menores a 100 Hz
    freq_limit = frequencies < 100
    return frequencies[freq_limit], magnitudes[freq_limit]

# Inicializar señal con ruido
noise = generate_noise(noise_50hz_init, noise_60hz_init, noise_baseline_init, 
                       noise_muscle_init, noise_white_init)
ecg_noisy = ecg_signal + noise

# Gráfica de ECG limpia
ax_ecg_clean.set_title('Señal ECG Original (Sin Ruido)', fontsize=12, fontweight='bold')
line_clean, = ax_ecg_clean.plot(t, ecg_signal, 'g-', linewidth=1.5, label='ECG Limpia')
ax_ecg_clean.set_xlabel('Tiempo (s)')
ax_ecg_clean.set_ylabel('Amplitud (normalizada)')
ax_ecg_clean.grid(True, alpha=0.3)
ax_ecg_clean.legend(loc='upper right')
ax_ecg_clean.set_xlim(0, duration)
ax_ecg_clean.set_ylim(-4, 4)

# Gráfica de ECG con ruido
ax_ecg_noisy.set_title('Señal ECG con Ruido Añadido', fontsize=12, fontweight='bold')
line_noisy, = ax_ecg_noisy.plot(t, ecg_noisy, 'r-', linewidth=1.5, label='ECG con Ruido')
ax_ecg_noisy.set_xlabel('Tiempo (s)')
ax_ecg_noisy.set_ylabel('Amplitud')
ax_ecg_noisy.grid(True, alpha=0.3)
ax_ecg_noisy.legend(loc='upper right')
ax_ecg_noisy.set_xlim(0, duration)
ax_ecg_noisy.set_ylim(-4, 4)

# Gráfica de FFT comparativa
ax_fft.set_title('Espectro de Frecuencias (FFT)', fontsize=12, fontweight='bold')
frequencies_clean, magnitudes_clean = calculate_fft(ecg_signal, sampling_rate)
frequencies_noisy, magnitudes_noisy = calculate_fft(ecg_noisy, sampling_rate)

line_fft_clean, = ax_fft.plot(frequencies_clean, magnitudes_clean, 'g-', 
                               linewidth=2, label='ECG Limpia', alpha=0.7)
line_fft_noisy, = ax_fft.plot(frequencies_noisy, magnitudes_noisy, 'r-', 
                               linewidth=2, label='ECG con Ruido')
ax_fft.set_xlabel('Frecuencia (Hz)')
ax_fft.set_ylabel('Magnitud')
ax_fft.grid(True, alpha=0.3)
ax_fft.legend(loc='upper right')
ax_fft.set_xlim(0, 100)
ax_fft.set_yscale('log')
ax_fft.set_ylim(0.001, 10)

# Crear sliders en la parte inferior
slider_ax_50hz = plt.axes([0.15, 0.20, 0.3, 0.02])
slider_ax_60hz = plt.axes([0.15, 0.17, 0.3, 0.02])
slider_ax_baseline = plt.axes([0.15, 0.14, 0.3, 0.02])
slider_ax_muscle = plt.axes([0.15, 0.11, 0.3, 0.02])
slider_ax_white = plt.axes([0.15, 0.08, 0.3, 0.02])

slider_50hz = Slider(slider_ax_50hz, 'Ruido 50 Hz', 0, 2, valinit=noise_50hz_init, 
                     valstep=0.05, color='orange')
slider_60hz = Slider(slider_ax_60hz, 'Ruido 60 Hz', 0, 2, valinit=noise_60hz_init, 
                     valstep=0.05, color='red')
slider_baseline = Slider(slider_ax_baseline, 'Deriva (0.5 Hz)', 0, 2, valinit=noise_baseline_init, 
                         valstep=0.05, color='blue')
slider_muscle = Slider(slider_ax_muscle, 'Ruido Muscular', 0, 1, valinit=noise_muscle_init, 
                       valstep=0.05, color='purple')
slider_white = Slider(slider_ax_white, 'Ruido Blanco', 0, 0.5, valinit=noise_white_init, 
                      valstep=0.01, color='gray')

# Crear botones
button_ax_play = plt.axes([0.60, 0.14, 0.12, 0.04])
button_ax_reset = plt.axes([0.60, 0.09, 0.12, 0.04])
button_play = Button(button_ax_play, 'Play/Pausa', color='lightblue', hovercolor='skyblue')
button_reset = Button(button_ax_reset, 'Reiniciar', color='lightcoral', hovercolor='salmon')

# Función para actualizar las gráficas
def update(val):
    amp_50 = slider_50hz.val
    amp_60 = slider_60hz.val
    amp_base = slider_baseline.val
    amp_musc = slider_muscle.val
    amp_wht = slider_white.val
    
    noise = generate_noise(amp_50, amp_60, amp_base, amp_musc, amp_wht)
    ecg_noisy = ecg_signal + noise
    
    line_noisy.set_ydata(ecg_noisy)
    
    # Actualizar FFT
    frequencies_noisy, magnitudes_noisy = calculate_fft(ecg_noisy, sampling_rate)
    line_fft_noisy.set_xdata(frequencies_noisy)
    line_fft_noisy.set_ydata(magnitudes_noisy)
    
    # Ajustar límites de Y si es necesario
    max_mag = max(np.max(magnitudes_noisy), np.max(magnitudes_clean))
    ax_fft.set_ylim(0.001, max(10, max_mag * 1.2))
    
    # Ajustar límites de señal con ruido
    y_max = max(4, np.max(np.abs(ecg_noisy)) * 1.1)
    ax_ecg_noisy.set_ylim(-y_max, y_max)
    
    fig.canvas.draw_idle()

# Función para animación
anim = None
phase = [0]

def toggle_animation(event):
    global is_playing, anim
    is_playing = not is_playing
    if is_playing:
        if anim is None:
            anim = FuncAnimation(fig, animate, interval=100, blit=False)
        button_play.label.set_text('Pausar')
    else:
        if anim is not None:
            anim.event_source.stop()
        button_play.label.set_text('Play')
    fig.canvas.draw_idle()

def animate(frame):
    if not is_playing:
        return
    
    phase[0] += 0.2
    
    amp_50 = slider_50hz.val
    amp_60 = slider_60hz.val
    amp_base = slider_baseline.val
    amp_musc = slider_muscle.val
    amp_wht = slider_white.val
    
    # Regenerar ruido con fase variable para animación
    noise_50 = amp_50 * np.sin(2 * np.pi * 50 * t + phase[0])
    noise_60 = amp_60 * np.sin(2 * np.pi * 60 * t + phase[0] * 1.2)
    noise_base = amp_base * np.sin(2 * np.pi * 0.5 * t + phase[0] * 0.1)
    noise_musc = amp_musc * (
        np.sin(2 * np.pi * 25 * t + phase[0]) + 
        0.5 * np.sin(2 * np.pi * 30 * t + phase[0] * 1.3) +
        0.3 * np.sin(2 * np.pi * 35 * t + phase[0] * 0.8)
    )
    noise_wht = amp_wht * np.random.randn(len(t))
    
    noise = noise_50 + noise_60 + noise_base + noise_musc + noise_wht
    ecg_noisy = ecg_signal + noise
    
    line_noisy.set_ydata(ecg_noisy)
    
    # Actualizar FFT
    frequencies_noisy, magnitudes_noisy = calculate_fft(ecg_noisy, sampling_rate)
    line_fft_noisy.set_xdata(frequencies_noisy)
    line_fft_noisy.set_ydata(magnitudes_noisy)

# Función para reiniciar
def reset(event):
    global is_playing, phase
    is_playing = False
    phase[0] = 0
    slider_50hz.reset()
    slider_60hz.reset()
    slider_baseline.reset()
    slider_muscle.reset()
    slider_white.reset()
    button_play.label.set_text('Play')
    if anim is not None:
        anim.event_source.stop()

# Conectar eventos
slider_50hz.on_changed(update)
slider_60hz.on_changed(update)
slider_baseline.on_changed(update)
slider_muscle.on_changed(update)
slider_white.on_changed(update)
button_play.on_clicked(toggle_animation)
button_reset.on_clicked(reset)

plt.show()
