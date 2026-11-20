import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. Cargar datos y filtrar nulos
df = pd.read_csv('/home/deigo/Descargas/summary/dataSummarySD1500.csv')
subset = df[['sd_energy', 'sd_s1000']].dropna()
x = subset['sd_s1000']
y = subset['sd_energy']

# 2. Ajuste lineal por mínimos cuadrados (y = mx + b)
m, b = np.polyfit(x, y, 1)

# 3. Elegir un valor arbitrario de x y calcular y con su propagación de error
x_arb = 30.0
y_arb = m * x_arb + b

# Calcular el error estándar de la regresión (desviación de los residuos)
y_pred = m * x + b
residual_std = np.std(y - y_pred)

print(f"Ecuación de la recta: y = {m:.4f}x + {b:.4f}")
print(f"Para x = {x_arb} -> y = {y_arb:.2f} ± {residual_std:.2f}")

# 4. Generar la gráfica final
plt.figure(figsize=(9, 6))
plt.scatter(x, y, alpha=0.3, color='gray', label='Datos (SD1500)')

# Trazar la línea recta de ajuste
x_vals = np.linspace(x.min(), x.max(), 100)
plt.plot(x_vals, m * x_vals + b, 'b-', linewidth=2, label=f'Ajuste lineal: y = {m:.3f}x + {b:.3f}')

# Graficar el punto arbitrario con su barra de error de propagación
plt.errorbar(x_arb, y_arb, yerr=residual_std, fmt='ro', markersize=8, capsize=5, 
             label=f'Punto arbitrario x={x_arb}\ny = {y_arb:.2f} ± {residual_std:.2f}')

plt.title('Parametrización de la Línea Recta y Propagación del Error', fontsize=12)
plt.xlabel('Señal a 1000m ($X$ = sd_s1000)', fontsize=11)
plt.ylabel('Energía [EeV] ($Y$ = sd_energy)', fontsize=11)
plt.legend(fontsize=10, loc='upper left')
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()