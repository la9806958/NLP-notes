import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Read the CSV file
df = pd.read_csv('parameter_sweep_timing_forward_ewma_40h_sp.csv')

# Get unique values for axes
k_factors = sorted(df['k_factors'].unique())
num_factors = sorted(df['num_factors'].unique())

# Create meshgrid for 3D plotting
K, N = np.meshgrid(k_factors, num_factors)

# Create pivot tables for each metric
formulation_pivot = df.pivot(index='num_factors', columns='k_factors', values='formulation_time_s')
solve_pivot = df.pivot(index='num_factors', columns='k_factors', values='solve_time_s')
total_pivot = df.pivot(index='num_factors', columns='k_factors', values='total_time_s')

# Create figure with 3 subplots
fig = plt.figure(figsize=(18, 6))

# Plot 1: Formulation Time
ax1 = fig.add_subplot(131, projection='3d')
surf1 = ax1.plot_surface(K, N, formulation_pivot.values, cmap='viridis', edgecolor='none', alpha=0.8)
ax1.set_title('Formulation Time (sec)', fontsize=12, fontweight='bold')
ax1.set_xlabel('Number of SOC Factor Exposures')
ax1.set_ylabel('Number of Factor Neutral Constraints')
ax1.set_zlabel('Time (s)')
fig.colorbar(surf1, ax=ax1, shrink=0.5, aspect=10)

# Plot 2: Solve Time
ax2 = fig.add_subplot(132, projection='3d')
surf2 = ax2.plot_surface(K, N, solve_pivot.values, cmap='viridis', edgecolor='none', alpha=0.8)
ax2.set_title('Solve Time (sec)', fontsize=12, fontweight='bold')
ax2.set_xlabel('Number of SOC Factor Exposures')
ax2.set_ylabel('Number of Factor Neutral Constraints')
ax2.set_zlabel('Time (s)')
fig.colorbar(surf2, ax=ax2, shrink=0.5, aspect=10)

# Plot 3: Total Time
ax3 = fig.add_subplot(133, projection='3d')
surf3 = ax3.plot_surface(K, N, total_pivot.values, cmap='viridis', edgecolor='none', alpha=0.8)
ax3.set_title('Total Time (sec)', fontsize=12, fontweight='bold')
ax3.set_xlabel('Number of SOC Factor Exposures')
ax3.set_ylabel('Number of Factor Neutral Constraints')
ax3.set_zlabel('Time (s)')
fig.colorbar(surf3, ax=ax3, shrink=0.5, aspect=10)

plt.suptitle('Parameter Sweep Timing Analysis', fontsize=14, fontweight='bold')
plt.tight_layout()

# Save the plot
plt.savefig('timing_3d_surface.png', dpi=150, bbox_inches='tight')
print("Plot saved to timing_3d_surface.png")

# Also generate heatmap version
formulation_pivot_hm = df.pivot(index='k_factors', columns='num_factors', values='formulation_time_s')
solve_pivot_hm = df.pivot(index='k_factors', columns='num_factors', values='solve_time_s')
total_pivot_hm = df.pivot(index='k_factors', columns='num_factors', values='total_time_s')

fig2, axes = plt.subplots(1, 3, figsize=(18, 6))

# Plot 1: Formulation Time
im1 = axes[0].imshow(formulation_pivot_hm.values, cmap='viridis', aspect='auto')
axes[0].set_title('Formulation Time (sec)', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Number of Factor Neutral Constraints')
axes[0].set_ylabel('Number of SOC Factor Exposures')
axes[0].set_xticks(range(len(num_factors)))
axes[0].set_xticklabels(num_factors)
axes[0].set_yticks(range(len(k_factors)))
axes[0].set_yticklabels(k_factors)
# Add text annotations
for i in range(len(k_factors)):
    for j in range(len(num_factors)):
        val = formulation_pivot_hm.values[i, j]
        text = f'{val:.2g}'
        axes[0].text(j, i, text, ha='center', va='center', fontsize=7, color='white' if val < 0.09 else 'black')
cbar1 = plt.colorbar(im1, ax=axes[0])
cbar1.set_label('Time (s)')

# Plot 2: Solve Time
im2 = axes[1].imshow(solve_pivot_hm.values, cmap='viridis', aspect='auto')
axes[1].set_title('Solve Time (sec)', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Number of Factor Neutral Constraints')
axes[1].set_ylabel('Number of SOC Factor Exposures')
axes[1].set_xticks(range(len(num_factors)))
axes[1].set_xticklabels(num_factors)
axes[1].set_yticks(range(len(k_factors)))
axes[1].set_yticklabels(k_factors)
# Add text annotations
for i in range(len(k_factors)):
    for j in range(len(num_factors)):
        val = solve_pivot_hm.values[i, j]
        text = f'{val:.2g}'
        axes[1].text(j, i, text, ha='center', va='center', fontsize=7, color='white' if val < 0.45 else 'black')
cbar2 = plt.colorbar(im2, ax=axes[1])
cbar2.set_label('Time (s)')

# Plot 3: Total Time
im3 = axes[2].imshow(total_pivot_hm.values, cmap='viridis', aspect='auto')
axes[2].set_title('Total Time (sec)', fontsize=12, fontweight='bold')
axes[2].set_xlabel('Number of Factor Neutral Constraints')
axes[2].set_ylabel('Number of SOC Factor Exposures')
axes[2].set_xticks(range(len(num_factors)))
axes[2].set_xticklabels(num_factors)
axes[2].set_yticks(range(len(k_factors)))
axes[2].set_yticklabels(k_factors)
# Add text annotations
for i in range(len(k_factors)):
    for j in range(len(num_factors)):
        val = total_pivot_hm.values[i, j]
        text = f'{val:.2g}'
        axes[2].text(j, i, text, ha='center', va='center', fontsize=7, color='white' if val < 0.5 else 'black')
cbar3 = plt.colorbar(im3, ax=axes[2])
cbar3.set_label('Time (s)')

plt.suptitle('Parameter Sweep Timing Analysis', fontsize=14, fontweight='bold')
plt.tight_layout()

plt.savefig('timing_heatmaps.png', dpi=150, bbox_inches='tight')
print("Plot saved to timing_heatmaps.png")

plt.show()
