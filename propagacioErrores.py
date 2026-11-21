from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Ruta a tus datos 
RUTA_CSV = Path("/home/deigo/Descargas/summary/dataSummarySD1500.csv")
SALIDA_IMG = Path("/home/deigo/Descargas/summary/propagacion_error_sd1500.png")

VAR_X = "sd_s1000"
VAR_Y = "sd_energy"

ANCHO_BIN = 25.0
X_EVAL = 30.0  # Valor arbitrario de sd_s1000 para propagar el error


def ajustar_y_propagar(x, y, y_err=None, nombre=""):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if y_err is None:
        # Mínimos cuadrados ordinarios (observaciones originales)
        n = len(x)
        dof = n - 2
        m, b = np.polyfit(x, y, 1)
        residuos = y - (m * x + b)
        s2 = float(np.sum(residuos ** 2) / dof) if dof > 0 else np.nan
        x_mean = float(np.mean(x))
        s_xx = float(np.sum((x - x_mean) ** 2))

        dm = float(np.sqrt(s2 / s_xx))
        db = float(np.sqrt(s2 * (1.0 / n + (x_mean ** 2) / s_xx)))
        cov_mb = float(-s2 * x_mean / s_xx)
    else:
        # Mínimos cuadrados ponderados (medias por bin con error std_y)
        y_err = np.asarray(y_err, dtype=float)
        mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(y_err) & (y_err > 0)
        x, y, y_err = x[mask], y[mask], y_err[mask]

        w = 1.0 / (y_err ** 2)
        W = float(np.sum(w))
        wx = float(np.sum(w * x))
        wy = float(np.sum(w * y))
        wxx = float(np.sum(w * x * x))
        wxy = float(np.sum(w * x * y))

        delta = W * wxx - wx ** 2
        b = float((wxx * wy - wx * wxy) / delta)
        m = float((W * wxy - wx * wy) / delta)

        db = float(np.sqrt(wxx / delta))
        dm = float(np.sqrt(W / delta))
        cov_mb = float(-wx / delta)

    def y_y_err(x_val):
        x_arr = np.asarray(x_val, dtype=float)
        y_pred = m * x_arr + b
        # Propagación: sigma_y^2 = (x * sigma_m)^2 + sigma_b^2 + 2 * x * cov(m, b)
        var_y = (x_arr * dm) ** 2 + (db ** 2) + 2.0 * x_arr * cov_mb
        sigma_y = np.sqrt(np.maximum(var_y, 0.0))
        return y_pred, sigma_y

    y_punto, err_punto = y_y_err(X_EVAL)
    print(f"\n=== {nombre} ===")
    print(f"Recta: y = {m:.6f}x + {b:.6f}")
    print(f"Incertidumbres: dm = {dm:.6f}, db = {db:.6f}, cov(m,b) = {cov_mb:.6e}")
    print(f"En X = {X_EVAL:.3f} -> y = {y_punto:.6f} ± {err_punto:.6f}")

    return {
        "m": m,
        "b": b,
        "dm": dm,
        "db": db,
        "cov_mb": cov_mb,
        "eval_fn": y_y_err,
        "y_punto": float(y_punto),
        "err_punto": float(err_punto),
    }


def main():
    df = pd.read_csv(RUTA_CSV)
    paired = (
        df[[VAR_X, VAR_Y]]
        .apply(pd.to_numeric, errors="coerce")
        .dropna()
    )
    paired = paired[np.isfinite(paired[VAR_X]) & np.isfinite(paired[VAR_Y])]

    x = paired[VAR_X].to_numpy(dtype=float)
    y = paired[VAR_Y].to_numpy(dtype=float)

    # 1. Propagación con ajuste en observaciones originales
    prop_orig = ajustar_y_propagar(x, y, nombre="Datos originales")

    # 2. Propagación con ajuste ponderado por bins
    edges = np.arange(x.min(), x.max() + ANCHO_BIN, ANCHO_BIN)
    filas_bins = []
    for i in range(len(edges) - 1):
        sub = paired[(paired[VAR_X] >= edges[i]) & (paired[VAR_X] < edges[i + 1])]
        if len(sub) >= 2 and sub[VAR_Y].std(ddof=1) > 0:
            filas_bins.append({
                "mean_x": sub[VAR_X].mean(),
                "mean_y": sub[VAR_Y].mean(),
                "std_y": sub[VAR_Y].std(ddof=1),
            })
    bins_df = pd.DataFrame(filas_bins)
    prop_bins = ajustar_y_propagar(
        bins_df["mean_x"],
        bins_df["mean_y"],
        y_err=bins_df["std_y"],
        nombre="Medias por bin (ponderado)",
    )

    # Gráfica de bandas de incertidumbre propagada
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    ax.scatter(x, y, alpha=0.25, s=10, color="slategray", label="Datos originales")

    x_grid = np.linspace(x.min(), x.max(), 300)

    y_o, err_o = prop_orig["eval_fn"](x_grid)
    ax.plot(x_grid, y_o, color="crimson", lw=2, label="Ajuste originales")
    ax.fill_between(
        x_grid, y_o - err_o, y_o + err_o,
        color="crimson", alpha=0.18, label="Banda ±σ_y (originales)"
    )

    y_b, err_b = prop_bins["eval_fn"](x_grid)
    ax.plot(x_grid, y_b, color="navy", lw=2, linestyle="--", label="Ajuste ponderado (bins)")
    ax.fill_between(
        x_grid, y_b - err_b, y_b + err_b,
        color="royalblue", alpha=0.20, label="Banda ±σ_y (bins)"
    )

    # Punto evaluado en X_EVAL
    ax.errorbar(
        X_EVAL,
        prop_bins["y_punto"],
        yerr=prop_bins["err_punto"],
        fmt="o",
        color="black",
        ecolor="black",
        capsize=4,
        zorder=6,
        label=f"X={X_EVAL:.1f} (bins): {prop_bins['y_punto']:.2f} ± {prop_bins['err_punto']:.2f}",
    )

    ax.set_title("Propagación del error para y = m·sd_s1000 + b", fontsize=11)
    ax.set_xlabel("Señal a 1000m (sd_s1000)", fontsize=10)
    ax.set_ylabel("Energía [EeV] (sd_energy)", fontsize=10)
    ax.legend(fontsize=8.5)
    ax.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.savefig(SALIDA_IMG, dpi=200)
    plt.show()


if __name__ == "__main__":
    main()