import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from typing import Tuple, Optional


class ConvolutionAnimator:
    """
    Clase para visualizar la convolución discreta paso a paso mediante animación.
    
    Attributes:
        x (np.ndarray): Señal de entrada
        h (np.ndarray): Respuesta al impulso (filtro)
        y (np.ndarray): Señal de salida (convolución completa)
    """
    
    def __init__(self, signal: np.ndarray, impulse_response: np.ndarray, 
                 figsize: Tuple[int, int] = (14, 7)):
        """
        Inicializa el animador de convolución.
        
        Args:
            signal: Señal de entrada x[n]
            impulse_response: Respuesta al impulso h[n]
            figsize: Tamaño de la figura (ancho, alto)
        """
        self.x = signal
        self.h = impulse_response
        self.y = np.convolve(self.x, self.h)
        
        # Ejes de tiempo
        self.n_x = np.arange(len(self.x))
        self.n_h = np.arange(len(self.h))
        self.n_y = np.arange(len(self.y))
        
        # Configuración de la figura
        self.fig, self.ax = plt.subplots(figsize=figsize)
        self._configure_plot()
        
    def _configure_plot(self) -> None:
        """Configura los parámetros iniciales del gráfico."""
        y_max = max(np.max(self.y), np.max(self.x), np.max(self.h)) + 1
        self.ax.set_xlim(-0.5, len(self.y) + 0.5)
        self.ax.set_ylim(-0.5, y_max)
        self.ax.grid(True, alpha=0.3, linestyle='--')
        self.ax.set_xlabel('Índice n', fontsize=12, fontweight='bold')
        self.ax.set_ylabel('Amplitud', fontsize=12, fontweight='bold')
        
    def _compute_partial_convolution(self, step: int) -> Tuple[float, np.ndarray]:
        """
        Calcula la convolución parcial en un paso dado.
        
        Args:
            step: Índice del paso actual
            
        Returns:
            Tupla con (valor acumulado, array de productos individuales)
        """
        h_flipped = self.h[::-1]
        y_partial = 0
        products = []
        
        for k in range(len(self.h)):
            x_index = step - k
            if 0 <= x_index < len(self.x):
                prod = self.x[x_index] * h_flipped[k]
            else:
                prod = 0
            products.append(prod)
            y_partial += prod
            
        return y_partial, np.array(products)
    
    def _animate_frame(self, step: int) -> None:
        """
        Actualiza el gráfico para un paso específico de la animación.
        
        Args:
            step: Índice del paso actual (0 a len(y)-1)
        """
        self.ax.clear()
        self._configure_plot()
        
        # Graficar señal de entrada x[n]
        self.ax.stem(self.n_x, self.x, linefmt='C0-', markerfmt='o', 
                     basefmt=' ', label='x[n] (entrada)')
        
        # Graficar h[n] volteada y desplazada
        h_flipped = self.h[::-1]
        self.ax.stem(self.n_h + step, h_flipped, linefmt='C3-', 
                     markerfmt='s', basefmt=' ', 
                     label=f'h[{step}-k] (filtro volteado)')
        
        # Calcular y visualizar productos
        y_partial, products = self._compute_partial_convolution(step)
        
        # Mostrar productos como barras verticales
        for k, prod in enumerate(products):
            x_pos = step - k
            if 0 <= x_pos < len(self.x) and prod > 0:
                self.ax.vlines(x_pos, 0, prod, colors='green', 
                              linewidth=8, alpha=0.6, 
                              label='Producto' if k == 0 else '')
        
        # Mostrar resultado acumulado y[n]
        y_computed = self.y[:step + 1]
        n_computed = self.n_y[:step + 1]
        self.ax.stem(n_computed, y_computed, linefmt='C5-', 
                     markerfmt='D', basefmt=' ', 
                     label='y[n] (salida)')
        
        # Título con información del paso actual
        self.ax.set_title(
            f'Convolución Discreta - Paso {step + 1}/{len(self.y)}\n'
            f'y[{step}] = Σ x[k]·h[{step}-k] = {y_partial:.2f}',
            fontsize=14, fontweight='bold', pad=20
        )
        
        # Leyenda sin duplicados
        handles, labels = self.ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        self.ax.legend(by_label.values(), by_label.keys(), 
                      loc='upper right', framealpha=0.9)
        
    def animate(self, interval: int = 800, repeat: bool = False, 
                save_path: Optional[str] = None) -> FuncAnimation:
        """
        Crea y muestra la animación de la convolución.
        
        Args:
            interval: Tiempo entre frames en milisegundos
            repeat: Si la animación debe repetirse
            save_path: Ruta para guardar la animación (opcional)
            
        Returns:
            Objeto FuncAnimation
        """
        anim = FuncAnimation(
            self.fig, 
            self._animate_frame, 
            frames=len(self.y),
            interval=interval, 
            repeat=repeat,
            blit=False
        )
        
        if save_path:
            anim.save(save_path, writer='pillow', fps=1)
            print(f"Animación guardada en: {save_path}")
            
        return anim


def create_square_wave(high_duration: int = 3, low_duration: int = 5, 
                       num_periods: int = 2) -> np.ndarray:
    """
    Crea una onda cuadrada periódica.
    
    Args:
        high_duration: Duración del nivel alto
        low_duration: Duración del nivel bajo
        num_periods: Número de períodos a generar
        
    Returns:
        Array con la onda cuadrada
    """
    single_period = np.concatenate([
        np.ones(high_duration),
        np.zeros(low_duration)
    ])
    return np.tile(single_period, num_periods)


def main():
    """Función principal para ejecutar la visualización."""
    # Definir señales
    x = create_square_wave(high_duration=3, low_duration=5, num_periods=2)
    h = np.ones(4)  # Pulso rectangular de longitud 4
    
    # Crear animador y ejecutar
    animator = ConvolutionAnimator(x, h, figsize=(14, 7))
    animation = animator.animate(interval=800, repeat=False)
    
    # Opcional: guardar animación
    # animation = animator.animate(interval=800, save_path='convolution.gif')
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()