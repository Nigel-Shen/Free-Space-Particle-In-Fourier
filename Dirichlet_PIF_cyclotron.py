import numpy as np
import matplotlib.pyplot as plt
import scipy.special as sp
import finufft
import pandas as pd
import matplotlib.animation as animation
import seaborn as sns

# Set font family for plots
plt.rcParams["font.family"] = "Times"

# Define a colormap
colormap = sns.cubehelix_palette(start=.5, rot=-.5, as_cmap=True, reverse=True)

# Create a figure and axis for plotting
fig, ax = plt.subplots(dpi=200)
artist = []

# Constants and parameters
rhos = []  # Stores /rho in physical space
L = 1
NG = 128
QM = -1
N = 40000
WP = 1  # omega p
VT = 1  # Thermal Velocity
lambdaD = VT / WP  # Debye length
dx = L / NG
B0 = np.array([0, 0, 300])
sigmas = np.array([[1 / 10], [1 / 30]]) / np.sqrt(2)

# Prepare for convolution kernel
extension = 4
wm = np.linspace(-NG * np.pi / L, NG * np.pi / L, extension * NG, endpoint=False)
wm1, wm2 = np.meshgrid(wm, wm)
s = np.sqrt(wm1 ** 2 + wm2 ** 2)

# Construct mollified Green's function
LT = 1.5 * L
green = (1 - sp.jv(0, LT * s)) / (s ** 2) - (LT * np.log(LT) * sp.jv(1, LT * s)) / s
green[extension * NG // 2, extension * NG // 2] = (LT ** 2 / 4 - LT ** 2 * np.log(LT) / 2)

# ... (continued comments for the rest of your code)

