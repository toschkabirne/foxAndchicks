#!/usr/bin/env python3
"""
Plot performance vs complexity for arena benchmark (appendix figure).
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from io import StringIO

ARENA_DATA = """Generation,Rank,Avg_Kills,Num_Neurons,Num_Connections
5000,1,0.0,28,19
5000,2,0.0,30,26
5000,3,18.0,28,17
5000,4,8.0,30,27
5000,5,15.8,28,14
10000,1,26.2,29,22
10000,2,25.4,31,26
10000,3,4.6,29,24
10000,4,17.4,32,29
10000,5,19.8,31,28
15000,1,18.6,30,31
15000,2,21.6,32,39
15000,3,27.8,32,37
15000,4,11.4,32,42
15000,5,19.8,32,41
20000,1,21.6,37,54
20000,2,21.6,31,43
20000,3,20.4,33,45
20000,4,18.4,33,45
20000,5,15.6,36,52
25000,1,23.4,32,50
25000,2,15.0,37,61
25000,3,14.0,34,48
25000,4,2.4,34,51
25000,5,17.4,34,56
30000,1,0.0,35,71
30000,2,23.0,34,53
30000,3,30.4,36,65
30000,4,29.2,36,65
30000,5,20.4,38,68
35000,1,31.4,35,67
35000,2,27.2,36,72
35000,3,20.0,35,66
35000,4,21.8,35,68
35000,5,9.0,37,68
40000,1,22.6,37,76
40000,2,3.6,39,76
40000,3,24.2,35,66
40000,4,32.8,35,69
40000,5,16.6,37,75
45000,1,25.6,37,74
45000,2,6.6,41,89
45000,3,32.0,37,74
45000,4,34.6,38,76
45000,5,19.2,36,80
50000,1,18.0,42,93
50000,2,3.2,42,106
50000,3,0.0,44,99
50000,4,35.2,39,82
50000,5,9.2,38,84
55000,1,16.4,46,114
55000,2,37.4,39,93
55000,3,36.0,42,92
55000,4,4.6,47,117
55000,5,28.2,44,101
60000,1,20.0,43,101
60000,2,31.4,43,105
60000,3,30.6,43,105
60000,4,31.6,43,106
60000,5,21.8,42,94
65000,1,41.4,44,109
65000,2,41.8,46,113
65000,3,40.0,43,107
65000,4,39.0,42,108
65000,5,34.4,45,112
70000,1,40.4,45,118
70000,2,38.4,45,117
70000,3,38.2,45,121
70000,4,31.2,45,118
70000,5,46.8,47,128
75000,1,52.8,46,131
75000,2,59.4,47,129
75000,3,44.0,46,131
75000,4,69.2,46,135
75000,5,52.6,46,128
80000,1,52.0,47,147
80000,2,37.0,48,150
80000,3,54.0,47,145
80000,4,23.2,49,143
80000,5,68.8,48,135
85000,1,67.4,50,150
85000,2,49.8,48,140
85000,3,54.4,48,140
85000,4,50.2,49,140
85000,5,59.8,48,142
"""

df = pd.read_csv(StringIO(ARENA_DATA))
Path('plots').mkdir(exist_ok=True)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))

# Shared color normalization
norm = plt.Normalize(vmin=df['Generation'].min(), vmax=df['Generation'].max())

# Left: kills vs connections
sc1 = ax1.scatter(df['Num_Connections'], df['Avg_Kills'],
                  c=df['Generation'], cmap='plasma', norm=norm, s=50, alpha=0.8, edgecolors='k', linewidths=0.3)
ax1.set_xlabel(r'Connections $c$')
ax1.set_ylabel(r'Mean kills $\bar{k}$')
ax1.set_title('Hunting Performance\nvs. Connection Complexity')
ax1.grid(True, alpha=0.3)

# Right: kills vs neurons
sc2 = ax2.scatter(df['Num_Neurons'], df['Avg_Kills'],
                  c=df['Generation'], cmap='plasma', norm=norm, s=50, alpha=0.8, edgecolors='k', linewidths=0.3)
ax2.set_xlabel(r'Neurons $n$')
ax2.set_title('Hunting Performance\nvs. Neuron Count')
ax2.grid(True, alpha=0.3)

# Single shared colorbar
cbar = fig.colorbar(sc2, ax=[ax1, ax2], location='right', shrink=0.85, pad=0.02)
cbar.set_label('Generation (ticks)')

plt.savefig('plots/arena_complexity.png', dpi=150, bbox_inches='tight')
print("Saved: plots/arena_complexity.png")
plt.show()
