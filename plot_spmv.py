# File: plot_spmv.py
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("spmv_benchmark.csv")

plt.figure(figsize=(10, 6))
for method in df['method'].unique():
    subset = df[df['method'] == method]
    plt.plot(subset['n'], subset['time_sec'], marker='o', label=method)

plt.xlabel("Matrix size (n)")
plt.ylabel("Execution Time (seconds)")
plt.title("SpMV Benchmark")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("spmv_benchmark_plot.png")
plt.show()
