import pandas as pd
import matplotlib.pyplot as plt

# 1. Cargar datos y filtrar valores nulos
df = pd.read_csv('/home/deigo/Descargas/summary/dataSummarySD1500.csv')
subset = df[['sd_energy', 'sd_s1000']].dropna()
x = subset['sd_s1000']
y = subset['sd_energy']

# 2. Generar la gráfica de dispersión
plt.figure(figsize=(8, 5))
plt.scatter(x, y, alpha=0.3, color='darkorange', s=15)
plt.title('Gráfica de Dispersión (sd_s1000 vs sd_energy)', fontsize=12)
plt.xlabel('Señal a 1000m (sd_s1000 - Variable X)', fontsize=10)
plt.ylabel('Energía [EeV] (sd_energy - Variable Y)', fontsize=10)
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()