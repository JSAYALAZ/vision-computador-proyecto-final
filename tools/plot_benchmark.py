"""
plot_benchmark.py — Gráficas para el informe (Parte I)
Proyecto Integrador Visión Artificial — Angélica Panamá, José Ayala

Lee los CSV que genera la app de escritorio y produce PNGs listos para el informe:
  benchmark_sweep.csv  (tecla 'b')  -> tiempo vs tamaño de kernel y speedup
  benchmark_live.csv   (siempre)    -> FPS y RAM por modo (CPU / GPU / Ambos)

Opcional: android_sweep.csv con las mismas columnas que benchmark_sweep.csv
(anoten a mano los ms que muestra la app para k = 3, 7, 11, ...).

Uso:
  python plot_benchmark.py --dir ../desktop/build --out ../docs/graficas
"""
import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd

CPU_COLOR = "#2a78d6"   # azul (serie 1)
GPU_COLOR = "#eb6834"   # naranja (serie 2)
INK = "#2b2b2b"
MUTED = "#8a8a85"
MODE_NAMES = {0: "CPU (cv::Mat)", 1: "GPU (cv::UMat)", 2: "Ambos"}


def style(ax, title, xlabel, ylabel):
    ax.set_title(title, loc="left", fontsize=12, color=INK, pad=10)
    ax.set_xlabel(xlabel, color=MUTED)
    ax.set_ylabel(ylabel, color=MUTED)
    ax.grid(axis="y", color="#e6e6e3", linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#c9c9c4")
    ax.tick_params(colors=MUTED)


def plot_sweep(df, out, tag):
    # 1) Tiempo por frame vs tamaño del kernel propio
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(df.kernel, df.cpu_ms, color=CPU_COLOR, lw=2, marker="o", ms=5, label="CPU (cv::Mat)")
    ax.plot(df.kernel, df.gpu_ms, color=GPU_COLOR, lw=2, marker="o", ms=5, label="GPU OpenCL (cv::UMat)")
    last = df.iloc[-1]
    ax.annotate(f"{last.cpu_ms:.1f} ms", (last.kernel, last.cpu_ms), xytext=(6, 0), textcoords="offset points",
                va="center", color=INK, fontsize=9)
    ax.annotate(f"{last.gpu_ms:.1f} ms", (last.kernel, last.gpu_ms), xytext=(6, 0), textcoords="offset points",
                va="center", color=INK, fontsize=9)
    style(ax, f"Tiempo de procesamiento por frame ({tag})", "Tamaño del kernel propio k (k×k)", "milisegundos")
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(out, f"{tag}_tiempo_vs_kernel.png"), dpi=200)
    plt.close(fig)

    # 2) Speedup = t_CPU / t_GPU
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(df.kernel, df.speedup, color=GPU_COLOR, lw=2, marker="o", ms=5)
    ax.axhline(1.0, color=MUTED, lw=1, ls="--")
    ax.text(df.kernel.min(), 1.02, "x1 (sin mejora)", color=MUTED, fontsize=8, va="bottom")
    style(ax, f"Aceleración de OpenCL sobre CPU ({tag})", "Tamaño del kernel propio k (k×k)", "speedup = t_CPU / t_GPU")
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(os.path.join(out, f"{tag}_speedup.png"), dpi=200)
    plt.close(fig)


def plot_live(df, out):
    df = df.copy()
    df["modo"] = df["mode"].map(MODE_NAMES)
    summary = df.groupby("modo").agg(fps=("fps", "mean"), ram=("ram_mb", "mean"),
                                     cpu_pct=("cpu_pct", "mean")).reindex(
        [m for m in MODE_NAMES.values() if m in set(df["modo"])])
    summary.to_csv(os.path.join(out, "resumen_por_modo.csv"))
    print(summary.round(2))

    for col, title, unit, fname in [("fps", "FPS promedio por modo", "cuadros por segundo", "fps_por_modo.png"),
                                    ("ram", "Memoria RAM del proceso por modo", "MB", "ram_por_modo.png")]:
        fig, ax = plt.subplots(figsize=(6, 3.6))
        colors = [CPU_COLOR if "CPU" in m else GPU_COLOR if "GPU" in m else "#1baf7a" for m in summary.index]
        bars = ax.bar(summary.index, summary[col], color=colors, width=0.55)
        for b, v in zip(bars, summary[col]):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.1f}", ha="center", va="bottom", color=INK, fontsize=9)
        style(ax, title, "", unit)
        fig.tight_layout()
        fig.savefig(os.path.join(out, fname), dpi=200)
        plt.close(fig)

    # FPS a lo largo del tiempo (se ve el efecto de mover el trackbar del kernel)
    fig, ax = plt.subplots(figsize=(8, 3.6))
    for m, color in [(0, CPU_COLOR), (1, GPU_COLOR), (2, "#1baf7a")]:
        sub = df[df["mode"] == m]
        if len(sub):
            ax.plot(sub.t_s, sub.fps, ".", ms=3, color=color, label=MODE_NAMES[m])
    style(ax, "FPS durante la sesión en vivo", "tiempo (s)", "FPS")
    ax.legend(frameon=False, markerscale=4)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fps_en_el_tiempo.png"), dpi=200)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=".", help="carpeta con los CSV")
    ap.add_argument("--out", default="graficas", help="carpeta de salida")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    for fname, tag in [("benchmark_sweep.csv", "escritorio"), ("android_sweep.csv", "android")]:
        p = os.path.join(a.dir, fname)
        if os.path.exists(p):
            plot_sweep(pd.read_csv(p), a.out, tag)
            print("OK", fname)
    p = os.path.join(a.dir, "benchmark_live.csv")
    if os.path.exists(p):
        plot_live(pd.read_csv(p), a.out)
        print("OK benchmark_live.csv")
    print("Gráficas en:", os.path.abspath(a.out))


if __name__ == "__main__":
    main()
