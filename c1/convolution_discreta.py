import numpy as np
import matplotlib.pyplot as plt

# Señales
x = np.array([2,2,4,5,8,2,1])
h = np.array([1,2])   # señal original
h_inv = h[::-1]       # inversión de h[k]

# Convolución
y = np.convolve(x, h, mode='full')

# Función auxiliar para anotar valores en la gráfica
def annotate_stem(ax, values):
    for i, v in enumerate(values):
        ax.text(i, v + 0.3, str(v), ha='center', fontsize=10, color="red")

plt.figure(figsize=(12,7))

# Señal x[k]
ax1 = plt.subplot(3,1,1)
ax1.stem(range(len(x)), x, basefmt=" ")
plt.title("Señal x[k]")
plt.xlabel("k")
plt.ylabel("x[k]")
annotate_stem(ax1, x)

# Señal h[k] invertida
ax2 = plt.subplot(3,1,2)
ax2.stem(range(len(h_inv)), h_inv, basefmt=" ")
plt.title("Señal h[k] invertida")
plt.xlabel("k")
plt.ylabel("h[k]")
annotate_stem(ax2, h_inv)

# Convolución / señal resultante
ax3 = plt.subplot(3,1,3)
ax3.stem(range(len(y)), y, basefmt=" ")
plt.title("Convolución y[k] = x[k] * h[k]")
plt.xlabel("k")
plt.ylabel("y[k]")
annotate_stem(ax3, y)

plt.tight_layout()
plt.show()