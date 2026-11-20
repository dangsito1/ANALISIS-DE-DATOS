import pandas as pd
import matplotlib.pyplot as plt

# 1. Cargar el archivo y extraer la variable de energía
df = pd.read_csv('/home/deigo/Descargas/summary/dataSummarySD1500.csv')
y = df['sd_energy'].dropna()

# 2. Generar el histograma con escala logarítmica
plt.figure(figsize=(8, 5))
plt.hist(y, bins=50, color='royalblue', edgecolor='black', alpha=0.7)

# Aplicar escala logarítmica en el eje Y
plt.yscale('log')

plt.title('Histograma de la variable sd_energy (Escala Logarítmica)', fontsize=12)
plt.xlabel('Energía [EeV]', fontsize=10)
plt.ylabel('Frecuencia (Log)', fontsize=10)
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()