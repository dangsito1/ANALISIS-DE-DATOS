import pandas as pd
import matplotlib.pyplot as plt

# 1. Cargar el archivo de datos y limpiar valores nulos
df = pd.read_csv('/home/deigo/Descargas/summary/dataSummarySD1500.csv')
subset = df[['sd_energy', 'sd_s1000']].dropna()
x = subset['sd_s1000']
y = subset['sd_energy']

# 2. Generar el histograma bivariado (2 variables aleatorias)
plt.figure(figsize=(9, 6))

# hist2d agrupa los datos en una malla de bines (aquí usamos 50x50)
# cmin=1 oculta las celdas vacías para que el fondo se vea limpio
hb = plt.hist2d(x, y, bins=50, cmap='viridis', cmin=1)

# Agregar barra de colores para indicar la frecuencia de eventos
cb = plt.colorbar(hb[3])
cb.set_label('Número de eventos (Frecuencia)', fontsize=11)

# 3. Personalizar la gráfica
plt.title(' Histograma bivariado de la Señal vs Energía', fontsize=13)
plt.xlabel('Señal a 1000m ($X$ = sd_s1000)', fontsize=11)
plt.ylabel('Energía [EeV] ($Y$ = sd_energy)', fontsize=11)
plt.grid(True, linestyle='--', alpha=0.3)

plt.tight_layout()
plt.show()