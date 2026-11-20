from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

ruta_csv = Path("/home/deigo/Descargas/summary/dataSummarySD1500.csv")  #

try:
    df = pd.read_csv(ruta_csv)  #[cite: 2]

    # Filtrar valores nulos y no finitos en las columnas de interés
    subset = df[["sd_s1000", "sd_energy"]].dropna()  #[cite: 2]
    subset = subset[
        np.isfinite(subset["sd_s1000"]) & np.isfinite(subset["sd_energy"])
    ]  #[cite: 2]

    X = subset["sd_s1000"].values  #[cite: 2]
    Y = subset["sd_energy"].values  #[cite: 2]

except Exception as e:
    print(f"Error cargando {ruta_csv}: {e}")
    X, Y = np.array([]), np.array([])

if len(X) == 0 or len(Y) == 0:
    print("[!] No se pudieron extraer datos válidos.")
else:
    # PROCESAMIENTO POR INTERVALOS (BINNING) CON ERRORES EN X E Y
    num_bins = 15
    bins = np.linspace(X.min(), X.max(), num_bins + 1)

    bin_means_x = []
    bin_stds_x = []
    bin_means_y = []
    bin_stds_y = []

    for i in range(num_bins):
        if i == num_bins - 1:
            mask = (X >= bins[i]) & (X <= bins[i + 1])
        else:
            mask = (X >= bins[i]) & (X < bins[i + 1])

        if np.sum(mask) > 0:
            bin_means_x.append(np.mean(X[mask]))
            bin_stds_x.append(np.std(X[mask]))
            bin_means_y.append(np.mean(Y[mask]))
            bin_stds_y.append(np.std(Y[mask]))

    bin_means_x = np.array(bin_means_x)
    bin_stds_x = np.array(bin_stds_x)
    bin_means_y = np.array(bin_means_y)
    bin_stds_y = np.array(bin_stds_y)

    bin_stds_x = np.where(bin_stds_x == 0, 1e-5, bin_stds_x)
    bin_stds_y = np.where(bin_stds_y == 0, 1e-5, bin_stds_y)

    #  AJUSTE LINEAL PONDERADO (CURVE FIT)
    def modelo_lineal(x, m, b):
        return m * x + b

    popt, pcov = curve_fit(
        modelo_lineal,
        bin_means_x,
        bin_means_y,
        sigma=bin_stds_y,
        absolute_sigma=True,
    )
    m, b = popt

    error_m = np.sqrt(pcov[0, 0])
    error_b = np.sqrt(pcov[1, 1])

    # CÁLCULO DE MÉTRICAS ESTADÍSTICAS
    r_val = np.corrcoef(bin_means_x, bin_means_y)[0, 1]
    y_pred = modelo_lineal(bin_means_x, m, b)
    ss_res = np.sum((bin_means_y - y_pred) ** 2)
    ss_tot = np.sum((bin_means_y - np.mean(bin_means_y)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

    chi2 = np.sum(((bin_means_y - y_pred) / bin_stds_y) ** 2)
    grados_libertad = len(bin_means_x) - 2
    chi2_red = chi2 / grados_libertad if grados_libertad > 0 else 0

    # PROPAGACIÓN DE ERRORES DE LA RECTA
    x_ajuste = np.linspace(X.min(), X.max(), 200)
    y_ajuste = modelo_lineal(x_ajuste, m, b)

    error_y_propagado = np.sqrt(
        (x_ajuste * error_m) ** 2
        + (error_b) ** 2
        + 2 * x_ajuste * pcov[0, 1]
    )

    # GRAFICACIÓN CON ERRORES EN AMBOS EJES (X, Y)
    plt.figure(figsize=(10, 6))

    plt.scatter(
        X,
        Y,
        color="indigo",
        alpha=0.15,
        edgecolors="none",
        s=30,
        label="Observaciones originales",
    )

    plt.fill_between(
        x_ajuste,
        y_ajuste - error_y_propagado,
        y_ajuste + error_y_propagado,
        color="crimson",
        alpha=0.15,
        label="Incertidumbre de la media ajustada",
    )

    etiqueta_ajuste = (
        f"Ajuste lineal: y = {m:.3f}x + {b:.2f}\n"
        f"r = {r_val:.4f}\n"
        f"R² = {r_squared:.4f}\n"
        f"X²_red = {chi2_red:.3f}"
    )
    plt.plot(x_ajuste, y_ajuste, color="crimson", linewidth=2, label=etiqueta_ajuste)

    plt.errorbar(
        bin_means_x,
        bin_means_y,
        yerr=bin_stds_y,
        xerr=bin_stds_x,
        fmt="s",
        color="red",
        ecolor="red",
        elinewidth=1.5,
        capsize=4,
        markersize=6,
        label="Media ± desviación estándar (X, Y)",
    )

    plt.title(
        "Medias por intervalo: sd_s1000 vs sd_energy", fontsize=14
    )  #[cite: 2]
    plt.xlabel("Señal a 1000m (sd_s1000)", fontsize=12)  #[cite: 2]
    plt.ylabel("Energía (sd_energy)", fontsize=12)  #[cite: 2]
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="lower right", frameon=True, edgecolor="gray")
    plt.tight_layout()
    plt.show()