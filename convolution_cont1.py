import numpy as np
import matplotlib.pyplot as plt

# Tiempo
t = np.linspace(0, 10, 500)

# Señales
x = np.heaviside(t, 1)        # escalón unitario
h = np.exp(-t) * np.heaviside(t,1)  # exponencial

# Convolución
y = np.convolve(x, h, mode='full') * (t[1]-t[0])  # discretización de integral
t_y = np.linspace(0, 2*t[-1], len(y))

# Graficar
plt.figure(figsize=(12,4))
plt.plot(t, x, label='x(t) - escalón')
plt.plot(t, h, label='h(t) - exponencial')
plt.plot(t_y, y, label='y(t) = x*h')
plt.title('Convolución: Escalón * Exponencial')
plt.xlabel('t')
plt.ylabel('Amplitud')
plt.grid(True)
plt.legend()
plt.show()
