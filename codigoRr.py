from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr


FILE_PATH = Path("/home/deigo/Descargas/summary/dataSummarySD1500.csv")
OUTPUT_DIR = Path("/home/deigo/Descargas/summary/resultados_sd1500")

X_COLUMN = "sd_s1000"
Y_COLUMN = "sd_energy"

X_LABEL = "Señal a 1000m (sd_s1000)"
Y_LABEL = "Energía [EeV] (sd_energy)"

BIN_WIDTH = 25.0
X_FOR_PROPAGATION = 30.0
MOSTRAR_EN_PANTALLA = True


def finalizar_figura(fig, ruta_salida):
    plt.tight_layout()
    plt.savefig(ruta_salida, dpi=150)
    if MOSTRAR_EN_PANTALLA:
        plt.show()
    plt.close(fig)


def linear_fit_ols(x, y, label):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    dof = n - 2

    slope, intercept = np.polyfit(x, y, 1)
    y_fit = slope * x + intercept
    residuals = y - y_fit
    sse = float(np.sum(residuals ** 2))
    residual_std = float(np.sqrt(sse / dof)) if dof > 0 else np.nan

    s_xx = float(np.sum((x - np.mean(x)) ** 2))
    slope_stderr = residual_std / np.sqrt(s_xx) if s_xx > 0 else np.nan
    intercept_stderr = (
        residual_std * np.sqrt(1.0 / n + (np.mean(x) ** 2) / s_xx)
        if s_xx > 0
        else np.nan
    )

    r_val, _ = pearsonr(x, y)
    r2_val = float(r_val ** 2)

    return {
        "label": label,
        "n": n,
        "dof": dof,
        "slope": float(slope),
        "intercept": float(intercept),
        "slope_stderr": float(slope_stderr),
        "intercept_stderr": float(intercept_stderr),
        "pearson_r": float(r_val),
        "r_squared": r2_val,
        "sse": sse,
        "residual_std": residual_std,
        "chi2": None,
        "reduced_chi2": None,
        "fitted": y_fit,
        "residuals": residuals,
        "x": x,
    }


def linear_fit_weighted(x, y, y_err, label):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    y_err = np.asarray(y_err, dtype=float)

    mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(y_err) & (y_err > 0)
    x, y, y_err = x[mask], y[mask], y_err[mask]

    n = len(x)
    dof = n - 2
    w = 1.0 / (y_err ** 2)
    W = np.sum(w)
    wx = np.sum(w * x)
    wy = np.sum(w * y)
    wxx = np.sum(w * x * x)
    wxy = np.sum(w * x * y)

    delta = W * wxx - wx ** 2
    intercept = (wxx * wy - wx * wxy) / delta
    slope = (W * wxy - wx * wy) / delta

    intercept_stderr = np.sqrt(wxx / delta)
    slope_stderr = np.sqrt(W / delta)

    y_fit = slope * x + intercept
    residuals = y - y_fit
    sse = float(np.sum(residuals ** 2))
    residual_std = float(np.sqrt(sse / dof)) if dof > 0 else np.nan
    chi2 = float(np.sum(((y - y_fit) / y_err) ** 2))
    reduced_chi2 = float(chi2 / dof) if dof > 0 else np.nan

    r_val, _ = pearsonr(x, y)
    r2_val = float(r_val ** 2)

    return {
        "label": label,
        "n": n,
        "dof": dof,
        "slope": float(slope),
        "intercept": float(intercept),
        "slope_stderr": float(slope_stderr),
        "intercept_stderr": float(intercept_stderr),
        "pearson_r": float(r_val),
        "r_squared": r2_val,
        "sse": sse,
        "residual_std": residual_std,
        "chi2": chi2,
        "reduced_chi2": reduced_chi2,
        "fitted": y_fit,
        "residuals": residuals,
        "x": x,
        "cov_m_b": -wx / delta,
    }


def propagate_mean(fit, x_val):
    y_val = fit["slope"] * x_val + fit["intercept"]
    dm = fit["slope_stderr"]
    db = fit["intercept_stderr"]
    cov_mb = fit.get("cov_m_b", 0.0)

    var_y = (x_val * dm) ** 2 + (db ** 2) + 2.0 * x_val * cov_mb
    err_y = float(np.sqrt(max(var_y, 0.0)))

    return {
        "x": float(x_val),
        "y": float(y_val),
        "incertidumbre_media": err_y,
        "lower": float(y_val - err_y),
        "upper": float(y_val + err_y),
    }


def print_fit(fit, prop):
    print(f"\n--- {fit['label']} ---")
    print(f"N = {fit['n']}")
    print(f"Grados de libertad = {fit['dof']}")
    print(f"Pendiente = {fit['slope']:.6f} ± {fit['slope_stderr']:.6f}")
    print(f"Intersección = {fit['intercept']:.6f} ± {fit['intercept_stderr']:.6f}")
    print(f"Pearson r = {fit['pearson_r']:.6f}")
    print(f"R² = {fit['r_squared']:.6f}")
    print(f"SSE = {fit['sse']:.6f}")
    print(f"Desviación residual = {fit['residual_std']:.6f}")
    if fit["chi2"] is not None:
        print(f"Chi-cuadrada = {fit['chi2']:.6f}")
        print(f"Chi-cuadrada reducida (chi2/ndf) = {fit['reduced_chi2']:.6f}")
    print(f"\nPara X = {prop['x']:.3f}:")
    print(f"Media ajustada = {prop['y']:.6f} ± {prop['incertidumbre_media']:.6f}")


def fit_row(fit, pred, dataset_name):
    return {
        "conjunto": dataset_name,
        "N_puntos_ajuste": fit["n"],
        "pendiente": fit["slope"],
        "error_pendiente": fit["slope_stderr"],
        "intersección (b)": fit["intercept"],
        "error_b": fit["intercept_stderr"],
        "correlacion_pearson": fit["pearson_r"],
        "R2": fit["r_squared"],
        "SSE": fit["sse"],
        "desviacion_residual": fit["residual_std"],
        "grados_libertad_residuales": fit["dof"],
        "chi2": fit["chi2"],
        "chi2_reducida": fit["reduced_chi2"],
        "X_elegida": pred["x"],
        "Y_estimado": pred["y"],
        "incertidumbre_media_ajustada": pred["incertidumbre_media"],
        "limite_inferior_media": pred["lower"],
        "limite_superior_media": pred["upper"],
        "metodo": (
            "Mínimos cuadrados ponderados"
            if fit["chi2"] is not None
            else "Mínimos cuadrados ordinarios"
        ),
    }


def main():
    if not FILE_PATH.exists():
        raise FileNotFoundError(f"No se encontró el CSV: {FILE_PATH}")

    (OUTPUT_DIR / "histogramas").mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "graficas_estadisticas").mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "resultados").mkdir(parents=True, exist_ok=True)

    data = pd.read_csv(FILE_PATH)
    for col in (X_COLUMN, Y_COLUMN):
        if col not in data.columns:
            raise KeyError(f"No existe la columna requerida: {col}")

    paired = (
        data[[X_COLUMN, Y_COLUMN]]
        .apply(pd.to_numeric, errors="coerce")
        .dropna()
    )
    paired = paired[np.isfinite(paired[X_COLUMN]) & np.isfinite(paired[Y_COLUMN])]

    if len(paired) < 3:
        raise ValueError("Se necesitan al menos tres observaciones completas.")

    # Estadística descriptiva
    descriptive = pd.DataFrame({
        "variable": [X_COLUMN, Y_COLUMN],
        "n": [len(paired), len(paired)],
        "media": [paired[X_COLUMN].mean(), paired[Y_COLUMN].mean()],
        "desv_std": [paired[X_COLUMN].std(ddof=1), paired[Y_COLUMN].std(ddof=1)],
        "mediana": [paired[X_COLUMN].median(), paired[Y_COLUMN].median()],
        "min": [paired[X_COLUMN].min(), paired[Y_COLUMN].min()],
        "max": [paired[X_COLUMN].max(), paired[Y_COLUMN].max()],
    })
    print("\n=== ANÁLISIS DESCRIPTIVO ===")
    print(descriptive.to_string(index=False))

    # 1. Histogramas individuales
    for col, fname, title in [
        (X_COLUMN, "histograma_sd_s1000.png", "Distribución de señal a 1000m"),
        (Y_COLUMN, "histograma_sd_energy.png", "Distribución de energía"),
    ]:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.hist(paired[col], bins=30, color="royalblue", edgecolor="black", alpha=0.8)
        ax.set_title(title)
        ax.set_xlabel(col)
        ax.set_ylabel("Frecuencia")
        ax.grid(True, linestyle="--", alpha=0.4)
        finalizar_figura(fig, OUTPUT_DIR / "histogramas" / fname)

    # 2. Dispersión simple
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(paired[X_COLUMN], paired[Y_COLUMN], alpha=0.4, s=12, color="indigo")
    ax.set_title(f"{X_LABEL} vs {Y_LABEL}")
    ax.set_xlabel(X_LABEL)
    ax.set_ylabel(Y_LABEL)
    ax.grid(True, linestyle="--", alpha=0.4)
    finalizar_figura(fig, OUTPUT_DIR / "graficas_estadisticas" / "dispersion_s1000_energia.png")

    # 3. Ajuste a observaciones originales
    raw_fit = linear_fit_ols(paired[X_COLUMN], paired[Y_COLUMN], "Ajuste a observaciones originales")
    raw_pred = propagate_mean(raw_fit, X_FOR_PROPAGATION)
    print_fit(raw_fit, raw_pred)

    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.scatter(paired[X_COLUMN], paired[Y_COLUMN], alpha=0.35, s=12, label="Datos (SD1500)")
    x_line = np.linspace(paired[X_COLUMN].min(), paired[X_COLUMN].max(), 200)
    ax.plot(
        x_line,
        raw_fit["slope"] * x_line + raw_fit["intercept"],
        color="crimson",
        linewidth=2,
        label=f"Ajuste: y = {raw_fit['slope']:.3f}x + {raw_fit['intercept']:.3f}",
    )
    ax.set_title("Ajuste lineal a todas las observaciones")
    ax.set_xlabel(X_LABEL)
    ax.set_ylabel(Y_LABEL)
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)
    raw_plot = OUTPUT_DIR / "graficas_estadisticas" / "ajuste_lineal_datos_originales.png"
    finalizar_figura(fig, raw_plot)

    # 4. Análisis y ajuste ponderado por bins
    x_min, x_max = paired[X_COLUMN].min(), paired[X_COLUMN].max()
    edges = np.arange(x_min, x_max + BIN_WIDTH, BIN_WIDTH)
    bin_rows = []
    for i in range(len(edges) - 1):
        sub = paired[(paired[X_COLUMN] >= edges[i]) & (paired[X_COLUMN] < edges[i + 1])]
        if len(sub) >= 2 and sub[Y_COLUMN].std(ddof=1) > 0:
            bin_rows.append({
                "bin_left": edges[i],
                "bin_right": edges[i + 1],
                "n": len(sub),
                "mean_x": sub[X_COLUMN].mean(),
                "std_x": sub[X_COLUMN].std(ddof=1),
                "mean_y": sub[Y_COLUMN].mean(),
                "std_y": sub[Y_COLUMN].std(ddof=1),
            })

    bins_stats = pd.DataFrame(bin_rows)
    bins_fit = linear_fit_weighted(
        bins_stats["mean_x"],
        bins_stats["mean_y"],
        bins_stats["std_y"],
        "Ajuste ponderado a medias por bin",
    )
    bins_pred = propagate_mean(bins_fit, X_FOR_PROPAGATION)
    print_fit(bins_fit, bins_pred)

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.scatter(paired[X_COLUMN], paired[Y_COLUMN], alpha=0.25, s=10, color="slateblue", label="Observaciones")
    ax.errorbar(
        bins_stats["mean_x"],
        bins_stats["mean_y"],
        xerr=bins_stats["std_x"],
        yerr=bins_stats["std_y"],
        fmt="s",
        color="darkred",
        ecolor="salmon",
        capsize=3,
        label="Media ± desv. estándar por bin",
    )
    ax.plot(
        x_line,
        bins_fit["slope"] * x_line + bins_fit["intercept"],
        color="black",
        linewidth=2,
        label=f"Ajuste ponderado: y = {bins_fit['slope']:.3f}x + {bins_fit['intercept']:.3f}",
    )
    ax.set_title("Medias por intervalo: media ± desviación estándar")
    ax.set_xlabel(X_LABEL)
    ax.set_ylabel(Y_LABEL)
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)
    bins_plot = OUTPUT_DIR / "graficas_estadisticas" / "medias_por_bins_y_ajuste.png"
    finalizar_figura(fig, bins_plot)

    # Preparar tablas para Excel
    comparison = pd.DataFrame([
        fit_row(raw_fit, raw_pred, "Datos originales"),
        fit_row(bins_fit, bins_pred, "Medias por bin"),
    ])

    raw_details = paired.copy()
    raw_details["y_ajustada"] = raw_fit["fitted"]
    raw_details["residuo"] = raw_fit["residuals"]

    bins_details = bins_stats.copy()
    bins_details["y_ajustada"] = bins_fit["slope"] * bins_details["mean_x"] + bins_fit["intercept"]
    bins_details["residuo_media"] = bins_details["mean_y"] - bins_details["y_ajustada"]

    propagation_details = pd.DataFrame(
        [raw_pred, bins_pred],
        index=["Originales", "Medias_bins"],
    )
    propagation_details.index.name = "Conjunto"

    output_excel = OUTPUT_DIR / "resultados" / "analisis_completo.xlsx"
    with pd.ExcelWriter(output_excel, engine="openpyxl") as writer:
        descriptive.to_excel(writer, sheet_name="Estadistica_descriptiva", index=False)
        comparison.to_excel(writer, sheet_name="Comparacion_ajustes", index=False)
        raw_details.to_excel(writer, sheet_name="Ajuste_datos_originales", index=False)
        bins_details.to_excel(writer, sheet_name="Estadistica_por_bins", index=False)
        propagation_details.to_excel(writer, sheet_name="Propagacion_error")

    print("\n=== ARCHIVOS GENERADOS ===")
    print(f"Excel: {output_excel}")
    print(f"Gráfica ajuste original: {raw_plot}")
    print(f"Gráfica medias por bins: {bins_plot}")


if __name__ == "__main__":
    main()