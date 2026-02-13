import numpy as np
import matplotlib.pyplot as plt


def simulate_clipped_lv(
    alpha, beta, delta, gamma, prey_start, pred_start, prey_cap, pred_cap, dt, steps
):

    t = np.linspace(0, steps * dt, steps)
    prey_hist = np.zeros(steps)
    pred_hist = np.zeros(steps)

    prey = prey_start
    pred = pred_start

    for i in range(steps):
        prey_hist[i] = prey
        pred_hist[i] = pred

        # Derivatives
        d_prey = (alpha * prey) - (beta * prey * pred)
        d_pred = (delta * prey * pred) - (gamma * pred)

        # Euler Step
        prey += d_prey * dt
        pred += d_pred * dt

        # --- YOUR CLIPPING LOGIC ---
        if prey > prey_cap:
            prey = prey_cap
        if pred > pred_cap:
            pred = pred_cap

        # Prevent extinction for the sake of the plot
        prey = max(prey, 0)
        pred = max(pred, 0)

    return t, prey_hist, pred_hist


ALPHA = 0.8  # Prey birth rate
GAMMA = 0.3  # Predator death rate

BETA = 0.005  # Predation rate (0.8 / 0.005 = 160 Equilibrium Preds)
DELTA = 0.0005  # Pred Repro rate (0.3 / 0.0005 = 600 Equilibrium Prey)

DT = 0.05
STEPS = 5000

PREY_START = 450
PRED_START = 110

PREY_CAP = 800
PRED_CAP = 175

# 1. Run WITH limits
t_clip, prey_clip, pred_clip = simulate_clipped_lv(
    ALPHA, BETA, DELTA, GAMMA, PREY_START, PRED_START, PREY_CAP, PRED_CAP, DT, STEPS
)

# 2. Run WITHOUT limits (Standard)
t_std, prey_std, pred_std = simulate_clipped_lv(
    ALPHA, BETA, DELTA, GAMMA, PREY_START, PRED_START, 9999, 9999, DT, STEPS
)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

ax1.set_title("Population Dynamics (Clipped vs Standard)")
ax1.plot(t_std, prey_std, "g--", alpha=0.3, label="Prey (Standard)")
ax1.plot(t_std, pred_std, "r--", alpha=0.3, label="Predator (Standard)")
ax1.plot(t_clip, prey_clip, "g-", linewidth=2, label=f"Prey (Cap {PREY_CAP})")
ax1.plot(t_clip, pred_clip, "r-", linewidth=2, label=f"Predator (Cap {PRED_CAP})")
ax1.set_xlabel("Time")
ax1.set_ylabel("Population Count")
ax1.grid(True, alpha=0.3)
ax1.legend(loc="upper right")

ax2.set_title("Phase Space: The 'Box' Effect")
ax2.plot(prey_std, pred_std, "k--", alpha=0.2, label="Standard Cycle")
ax2.plot(prey_clip, pred_clip, "b-", linewidth=2, label="Clipped Trajectory")

# Draw the Limits
ax2.axvline(
    x=PREY_CAP,
    color="green",
    linestyle=":",
    linewidth=3,
    label=f"Prey Cap ({PREY_CAP})",
)
ax2.axhline(
    y=PRED_CAP,
    color="red",
    linestyle=":",
    linewidth=3,
    label=f"Predator Cap ({PRED_CAP})",
)

ax2.set_xlabel("Prey Population")
ax2.set_ylabel("Predator Population")
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
