import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# 1. Cargar datos y filtrar valores nulos
df = pd.read_csv('/home/deigo/Descargas/summary/dataSummarySD1500.csv')
subset = df[['sd_energy', 'sd_s1000']].dropna()
x = subset['sd_s1000']
y = subset['sd_energy']

# 2. Configurar la figura con subgráficas (GridSpec) para la dispersión y las proyecciones
fig = plt.figure(figsize=(9, 9))
gs = gridspec.GridSpec(2, 2, width_ratios=[4, 1], height_ratios=[1, 4], hspace=0.05, wspace=0.05)

ax_scatter = fig.add_subplot(gs[1, 0])
ax_hist_x = fig.add_subplot(gs[0, 0], sharex=ax_scatter)
ax_hist_y = fig.add_subplot(gs[1, 1], sharey=ax_scatter)

# Gráfica de dispersión central
ax_scatter.scatter(x, y, alpha=0.3, color='darkorange', s=12)
ax_scatter.set_xlabel('Señal a 1000m ($X$ = sd_s1000)', fontsize=11)
ax_scatter.set_ylabel('Energía [EeV] ($Y$ = sd_energy)', fontsize=11)
ax_scatter.grid(True, linestyle='--', alpha=0.5)

# Proyección en el eje X (Histograma superior)
ax_hist_x.hist(x, bins=50, color='royalblue', edgecolor='black', alpha=0.7)
ax_hist_x.set_title('Proyección de cada variable en su eje', fontsize=12, pad=10)
ax_hist_x.tick_params(labelbottom=False)
ax_hist_x.grid(True, linestyle='--', alpha=0.3)

# Proyección en el eje Y (Histograma derecho)
ax_hist_y.hist(y, bins=50, orientation='horizontal', color='forestgreen', edgecolor='black', alpha=0.7)
#ax_hist_y.set_yscale('log')
ax_hist_y.tick_params(labelleft=False)
ax_hist_y.grid(True, linestyle='--', alpha=0.3)

plt.show()