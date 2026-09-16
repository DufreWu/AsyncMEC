import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =====================================================
# Configuration
# =====================================================

LOG_DIR = "logs"

CONTROLLERS = [
    "max_perf",
    "adaptive",
    "multi_scale"
]

FPS_REQ = 25
CONTROL_PERIOD = 1.0  # second


# =====================================================
# Calculate Metrics
# =====================================================

def calculate_metrics(df):

    # -------------------------------------
    # Runtime
    # -------------------------------------

    avg_fps = df["fps"].mean()

    qos_sat = (
        (df["fps"] >= FPS_REQ)
        .mean()
        * 100
    )

    qos_violation = 100 - qos_sat

    # -------------------------------------
    # Power
    # -------------------------------------

    power = (
        df["comp_power"]
        + df["mech_power"]
    )

    avg_power = power.mean()

    # -------------------------------------
    # Energy
    # -------------------------------------

    total_energy_wh = (
        power.sum()
        * CONTROL_PERIOD
        / 3600.0
    )

    # -------------------------------------
    # Distance
    # -------------------------------------

    distance = (
        df["speed"]
        * CONTROL_PERIOD
    ).sum()

    # -------------------------------------
    # Cyber-Physical Efficiency
    # -------------------------------------

    eta_cp = (
        distance
        / total_energy_wh
    )

    # -------------------------------------
    # Battery Health
    # -------------------------------------

    avg_current = (
        df["current"]
        .mean()
    )

    rms_current = np.sqrt(
        np.mean(
            df["current"] ** 2
        )
    )

    avg_temp = (
        df["temperature"]
        .mean()
    )

    peak_temp = (
        df["temperature"]
        .max()
    )

    return {
        "FPS": avg_fps,
        "QoS(%)": qos_sat,
        "QoS_Violation(%)": qos_violation,

        "Energy(Wh)": total_energy_wh,
        "Power(W)": avg_power,

        "Distance(m)": distance,
        "EtaCP(m/Wh)": eta_cp,

        "Current(A)": avg_current,
        "IRMS(A)": rms_current,

        "Temp(C)": avg_temp,
        "PeakTemp(C)": peak_temp,
    }


# =====================================================
# Load Logs
# =====================================================

results = []

for controller in CONTROLLERS:

    file_path = os.path.join(
        LOG_DIR,
        f"{controller}.csv"
    )

    if not os.path.exists(file_path):

        print(
            f"WARNING: {file_path} not found"
        )
        continue

    print(
        f"Loading {file_path}"
    )

    df = pd.read_csv(file_path)

    metrics = calculate_metrics(df)

    metrics["Controller"] = controller

    results.append(metrics)

# =====================================================
# Summary Table
# =====================================================

summary = pd.DataFrame(results)

summary = summary[
    [
        "Controller",

        "FPS",
        "QoS(%)",

        "Energy(Wh)",
        "Power(W)",

        "Distance(m)",
        "EtaCP(m/Wh)",

        "Current(A)",
        "IRMS(A)",

        "Temp(C)",
        "PeakTemp(C)"
    ]
]

print("\n===================================")
print("SUMMARY")
print("===================================")

print(summary)

summary.to_csv(
    os.path.join(
        LOG_DIR,
        "summary.csv"
    ),
    index=False
)

print(
    "\nSaved logs/summary.csv"
)


# =====================================================
# Figure 1
# Runtime Performance
# =====================================================

plt.figure(figsize=(6,4))

plt.bar(
    summary["Controller"],
    summary["FPS"]
)

plt.ylabel("Average FPS")
plt.title("Runtime Performance")

plt.tight_layout()

plt.savefig(
    os.path.join(
        LOG_DIR,
        "runtime_performance.png"
    ),
    dpi=300
)

# =====================================================
# Figure 2
# Energy
# =====================================================

plt.figure(figsize=(6,4))

plt.bar(
    summary["Controller"],
    summary["Energy(Wh)"]
)

plt.ylabel("Energy (Wh)")
plt.title("Energy Consumption")

plt.tight_layout()

plt.savefig(
    os.path.join(
        LOG_DIR,
        "energy_consumption.png"
    ),
    dpi=300
)

# =====================================================
# Figure 3
# Battery Health
# =====================================================

plt.figure(figsize=(6,4))

plt.bar(
    summary["Controller"],
    summary["IRMS(A)"]
)

plt.ylabel("RMS Current (A)")
plt.title("Battery Stress")

plt.tight_layout()

plt.savefig(
    os.path.join(
        LOG_DIR,
        "battery_health.png"
    ),
    dpi=300
)

# =====================================================
# Figure 4
# Pareto
# QoS vs Energy
# =====================================================

plt.figure(figsize=(6,5))

for _, row in summary.iterrows():

    plt.scatter(
        row["Energy(Wh)"],
        row["FPS"],
        s=120
    )

    plt.annotate(
        row["Controller"],
        (
            row["Energy(Wh)"],
            row["FPS"]
        )
    )

plt.xlabel("Energy (Wh)")
plt.ylabel("QoS (FPS)")
plt.title("Pareto Front: QoS vs Energy")

plt.grid(True)

plt.tight_layout()

plt.savefig(
    os.path.join(
        LOG_DIR,
        "pareto_qos_energy.png"
    ),
    dpi=300
)

# =====================================================
# Figure 5
# Pareto
# Energy vs Battery Health
# =====================================================

plt.figure(figsize=(6,5))

for _, row in summary.iterrows():

    plt.scatter(
        row["Energy(Wh)"],
        row["IRMS(A)"],
        s=120
    )

    plt.annotate(
        row["Controller"],
        (
            row["Energy(Wh)"],
            row["IRMS(A)"]
        )
    )

plt.xlabel("Energy (Wh)")
plt.ylabel("RMS Current (A)")
plt.title(
    "Pareto Front: Energy vs Battery Health"
)

plt.grid(True)

plt.tight_layout()

plt.savefig(
    os.path.join(
        LOG_DIR,
        "pareto_energy_health.png"
    ),
    dpi=300
)

# =====================================================
# Per-Phase Statistics
# =====================================================

for controller in CONTROLLERS:

    file_path = os.path.join(
        LOG_DIR,
        f"{controller}.csv"
    )

    if not os.path.exists(file_path):
        continue

    df = pd.read_csv(file_path)

    phase_results = []

    for phase in df["phase"].unique():

        d = df[
            df["phase"] == phase
        ]

        power = (
            d["comp_power"]
            + d["mech_power"]
        )

        energy = (
            power.sum()
            * CONTROL_PERIOD
            / 3600
        )

        distance = (
            d["speed"]
            * CONTROL_PERIOD
        ).sum()

        phase_results.append(
            {
                "Phase": phase,
                "FPS": d["fps"].mean(),
                "Energy(Wh)": energy,
                "Distance(m)": distance,
                "IRMS(A)": np.sqrt(
                    np.mean(
                        d["current"]**2
                    )
                )
            }
        )

    phase_df = pd.DataFrame(
        phase_results
    )

    phase_df.to_csv(
        os.path.join(
            LOG_DIR,
            f"{controller}_phase.csv"
        ),
        index=False
    )

print("\nDone.")