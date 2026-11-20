from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ruta_csv = Path("/home/deigo/Descargas/summary/dataSummarySD1500.csv")  #[cite: 2]
df = pd.read_csv(ruta_csv)  #[cite: 2]

# Variables a comparar lado a lado (puedes cambiar los nombres por otras columnas)
var_izq = "sd_s1000"  #[cite: 2]
var_der = "sd_energy"  #

datos_izq = pd.to_numeric(df[var_izq], errors="coerce").dropna()
datos_izq = datos_izq[np.isfinite(datos_izq)]

datos_der = pd.to_numeric(df[var_der], errors="coerce").dropna()
datos_der = datos_der[np.isfinite(datos_der)]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

# Histograma izquierdo: sd_s1000
ax1.hist(
    datos_izq, bins=35, color="forestgreen", edgecolor="black", alpha=0.8
)
ax1.set_title("Histograma: Distribución de Señal a 1000m (sd_s1000)", fontsize=11)  #[cite: 2]
ax1.set_xlabel("Señal a 1000m (sd_s1000)", fontsize=10)  #[cite: 2]
ax1.set_ylabel("Frecuencia (Cantidad de eventos)", fontsize=10)
ax1.grid(True, linestyle="--", alpha=0.4)

# Histograma derecho: sd_energy
ax2.hist(
    datos_der, bins=35, color="cornflowerblue", edgecolor="black", alpha=0.85
)
ax2.set_title("Histograma: Distribución de Energía (sd_energy)", fontsize=11)  #[cite: 2, 6]
ax2.set_xlabel("Energía [EeV] (sd_energy)", fontsize=10)  #[cite: 2, 6]
ax2.set_ylabel("Frecuencia (Cantidad de eventos)", fontsize=10)
ax2.grid(True, linestyle="--", alpha=0.4)

plt.tight_layout()
plt.savefig("/home/deigo/Descargas/summary/histogramas_lado_a_lado.png", dpi=200)  #[cite: 2]
plt.show()