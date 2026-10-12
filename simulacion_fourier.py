import numpy as np
import matplotlib.pyplot as plt

# Frecuencia de muestreo y eje de tiempo
fs = 500
duracion = 2
n = int(fs * duracion)
t = np.arange(n) / fs - duracion / 2
dt = 1 / fs

# Señales elementales
pulso = (np.abs(t) <= 0.2).astype(float)
escalon = (t >= 0).astype(float)
senoidal = np.sin(2 * np.pi * 5 * t)

def espectro(signal):
    """Devuelve frecuencias, magnitud y fase de la FFT centrada."""
    fft_signal = np.fft.fftshift(np.fft.fft(signal))
    frecuencias = np.fft.fftshift(np.fft.fftfreq(len(signal), d=dt))

    magnitud = np.abs(fft_signal) / len(signal)
    fase = np.angle(fft_signal)

    return frecuencias, magnitud, fase

def graficar_espectro(ax_magnitud, ax_fase, signal, nombre):
    frecuencias, magnitud, fase = espectro(signal)

    ax_magnitud.plot(frecuencias, magnitud)
    ax_magnitud.set_title(f"Magnitud: {nombre}")
    ax_magnitud.set_xlabel("Frecuencia (Hz)")
    ax_magnitud.set_ylabel("Magnitud")
    ax_magnitud.set_xlim(-40, 40)
    ax_magnitud.grid(True)

    ax_fase.plot(frecuencias, fase)
    ax_fase.set_title(f"Fase: {nombre}")
    ax_fase.set_xlabel("Frecuencia (Hz)")
    ax_fase.set_ylabel("Fase (rad)")
    ax_fase.set_xlim(-40, 40)
    ax_fase.grid(True)

# Señales en el tiempo y sus espectros
senales = [
    ("Pulso rectangular", pulso),
    ("Función escalón", escalon),
    ("Señal senoidal de 5 Hz", senoidal),
]

fig, ejes = plt.subplots(3, 3, figsize=(15, 11))

for fila, (nombre, signal) in enumerate(senales):
    ejes[fila, 0].plot(t, signal)
    ejes[fila, 0].set_title(f"{nombre}: dominio del tiempo")
    ejes[fila, 0].set_xlabel("Tiempo (s)")
    ejes[fila, 0].set_ylabel("Amplitud")
    ejes[fila, 0].grid(True)

    graficar_espectro(ejes[fila, 1], ejes[fila, 2], signal, nombre)

fig.tight_layout()

# Propiedad 1: linealidad
# FFT(a*x + b*y) = a*FFT(x) + b*FFT(y)
x = pulso
y = senoidal
a = 2
b = 0.5

error_linealidad = np.max(np.abs(
    np.fft.fft(a * x + b * y)
    - (a * np.fft.fft(x) + b * np.fft.fft(y))
))

# Propiedad 2: desplazamiento temporal
# Desplazar una señal cambia la fase de su espectro.
desplazamiento_muestras = 50
pulso_desplazado = np.roll(pulso, desplazamiento_muestras)

# Propiedad 3: escalamiento temporal
# x(2t) está comprimida en el tiempo y su espectro se ensancha.
pulso_comprimido = (np.abs(2 * t) <= 0.2).astype(float)

fig2, ejes2 = plt.subplots(2, 2, figsize=(12, 8))

ejes2[0, 0].plot(t, pulso, label="Original")
ejes2[0, 0].plot(t, pulso_desplazado, "--", label="Desplazada")
ejes2[0, 0].set_title("Desplazamiento en el tiempo")
ejes2[0, 0].set_xlabel("Tiempo (s)")
ejes2[0, 0].set_ylabel("Amplitud")
ejes2[0, 0].legend()
ejes2[0, 0].grid(True)

f, mag_original, _ = espectro(pulso)
_, mag_desplazado, _ = espectro(pulso_desplazado)
ejes2[0, 1].plot(f, mag_original, label="Original")
ejes2[0, 1].plot(f, mag_desplazado, "--", label="Desplazada")
ejes2[0, 1].set_title("El desplazamiento conserva la magnitud")
ejes2[0, 1].set_xlabel("Frecuencia (Hz)")
ejes2[0, 1].set_ylabel("Magnitud")
ejes2[0, 1].set_xlim(-40, 40)
ejes2[0, 1].legend()
ejes2[0, 1].grid(True)

ejes2[1, 0].plot(t, pulso, label="Original")
ejes2[1, 0].plot(t, pulso_comprimido, "--", label="Comprimido")
ejes2[1, 0].set_title("Escalamiento en el tiempo")
ejes2[1, 0].set_xlabel("Tiempo (s)")
ejes2[1, 0].set_ylabel("Amplitud")
ejes2[1, 0].legend()
ejes2[1, 0].grid(True)

f, mag_original, _ = espectro(pulso)
_, mag_comprimido, _ = espectro(pulso_comprimido)
ejes2[1, 1].plot(f, mag_original, label="Original")
ejes2[1, 1].plot(f, mag_comprimido, "--", label="Comprimido")
ejes2[1, 1].set_title("La compresión ensancha el espectro")
ejes2[1, 1].set_xlabel("Frecuencia (Hz)")
ejes2[1, 1].set_ylabel("Magnitud")
ejes2[1, 1].set_xlim(-40, 40)
ejes2[1, 1].legend()
ejes2[1, 1].grid(True)

fig2.tight_layout()

print(f"Error numérico de la propiedad de linealidad: {error_linealidad:.2e}")
print("La fase puede cambiar al desplazar una señal, aunque su magnitud se conserve.")
print("Al comprimir una señal en el tiempo, su espectro se extiende en frecuencia.")

plt.show()