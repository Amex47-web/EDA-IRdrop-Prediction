import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Step 1: Load prediction
pred = np.load('/content/EDA-IRdrop-Prediction/output/PredArray.npy')
print("Prediction shape:", pred.shape)

# Step 2: Define function to compute region-wise averages
def compute_region_drops(pred, num_regions=8):
    h, w = pred.shape
    step_h, step_w = h // num_regions, w // num_regions
    region_drops = np.zeros((num_regions, num_regions))

    for i in range(num_regions):
        for j in range(num_regions):
            region = pred[i*step_h:(i+1)*step_h, j*step_w:(j+1)*step_w]
            region_drops[i, j] = np.mean(region)

    return region_drops

# Step 3: Compute IR drop per region
num_regions = 8  # You can change this to 4, 16, etc.
region_drops = compute_region_drops(pred, num_regions)

# Step 4: Show as heatmap
plt.figure(figsize=(6,5))
plt.imshow(region_drops, cmap='plasma', interpolation='nearest')
plt.colorbar(label='Average IR Drop (V)')
plt.title(f'Region-wise IR Drop Map ({num_regions}x{num_regions})')
plt.xlabel("Region X")
plt.ylabel("Region Y")
plt.show()

# Step 5: Convert to DataFrame for easier reading
df = pd.DataFrame(region_drops,
                  index=[f'Y{i+1}' for i in range(num_regions)],
                  columns=[f'X{j+1}' for j in range(num_regions)])
print("\nAverage IR Drop per Region (in Volts):")
print(df.round(6))


---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

num_regions = int(input("Enter number of grid regions (e.g., 4, 8, 16): "))
threshold = float(input("Enter IR-drop threshold (e.g., 0.05 for 50mV): "))

region_drops = compute_region_drops(pred, num_regions)

# Identify critical (hot) regions
critical_regions = np.argwhere(region_drops > threshold)
print(f"\n Critical Regions above {threshold} V drop:")
for (y, x) in critical_regions:
    print(f"  Region Y{y+1}, X{x+1} — Avg Drop = {region_drops[y, x]:.6f} V")

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D

#  Load prediction
pred = np.load('/content/EDA-IRdrop-Prediction/output/PredArray.npy')
print(f" Prediction loaded. Shape: {pred.shape}")

# Function to compute detailed region statistics
def compute_region_stats(pred, num_regions=8):
    h, w = pred.shape
    step_h, step_w = h // num_regions, w // num_regions

    data = []
    stats = np.zeros((num_regions, num_regions))

    for i in range(num_regions):
        for j in range(num_regions):
            region = pred[i*step_h:(i+1)*step_h, j*step_w:(j+1)*step_w]
            mean_val = np.mean(region)
            max_val = np.max(region)
            min_val = np.min(region)
            std_val = np.std(region)

            data.append({
                "Region_Y": i+1,
                "Region_X": j+1,
                "Mean_IRDrop": mean_val,
                "Max_IRDrop": max_val,
                "Min_IRDrop": min_val,
                "Std_Deviation": std_val
            })
            stats[i, j] = mean_val  # For heatmap

    return pd.DataFrame(data), stats

#  Compute region statistics
num_regions = 8
region_df, region_map = compute_region_stats(pred, num_regions)

# Heatmap visualization
plt.figure(figsize=(6,5))
plt.imshow(region_map, cmap='inferno', interpolation='bilinear')
plt.colorbar(label='Average IR Drop (V)')
plt.title(f'Region-wise Mean IR Drop ({num_regions}x{num_regions})')
plt.xlabel("Region X")
plt.ylabel("Region Y")
plt.show()

# 3D Surface Plot
X, Y = np.meshgrid(np.arange(num_regions), np.arange(num_regions))
fig = plt.figure(figsize=(7,5))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, region_map, cmap='plasma', edgecolor='none')
ax.set_title('3D IR Drop Surface')
ax.set_xlabel('Region X')
ax.set_ylabel('Region Y')
ax.set_zlabel('Avg IR Drop (V)')
plt.show()

# Display Ranked Regions by Mean IR Drop
sorted_df = region_df.sort_values(by="Mean_IRDrop", ascending=False)
print("\n Top 5 Critical Regions (Highest IR Drop):")
print(sorted_df.head(5).round(6))

# Save results to CSV
csv_path = '/content/EDA-IRdrop-Prediction/output/Region_IRDrop_Stats.csv'
region_df.to_csv(csv_path, index=False)
print(f"\n Detailed region stats saved to: {csv_path}")

#  Summary Metrics
overall_mean = np.mean(pred)
overall_max = np.max(pred)
overall_std = np.std(pred)

print("\n Overall IR Drop Summary:")
print(f"Mean IR Drop: {overall_mean:.6f} V")
print(f"Max IR Drop:  {overall_max:.6f} V")
print(f"Std Dev:      {overall_std:.6f} V")

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

import numpy as np

# Load base prediction (at 1.0V)
pred = np.load('/content/EDA-IRdrop-Prediction/output/PredArray.npy')

def simulate_ir_voltage_variation(pred, v_supply):
    base_v = 1.0  # because model trained at 1.0 V
    scaling_factor = base_v / v_supply
    noise = np.random.normal(0, 0.002, pred.shape)
    return np.clip(pred * scaling_factor + noise, 0, None)

voltages = [0.8, 0.9, 1.0, 1.1, 1.2]

base_mean = np.mean(pred)
for v in voltages:
    sim_pred = simulate_ir_voltage_variation(pred, v)
    delta_mv = (np.mean(sim_pred) - base_mean) * 1000
    print(f"Δ IR drop at {v} V: {delta_mv:.3f} mV")

