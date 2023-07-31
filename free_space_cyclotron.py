'''
MIT License

Copyright (c) 2023 Nigel Shen

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''

import numpy as np
import matplotlib.pyplot as plt
import scipy.special as sp
import finufft
import matplotlib.animation as animation
import pandas as pd

# Define constants and parameters
L = 1
NG = 128
N = 40000
QM = -1
WP = 1  # omega p
VT = 1  # Thermal Velocity
lambdaD = VT / WP
dx = L / NG
B0 = np.array([0, 0, 300])
sigmas = np.array([[1 / 10], [1 / 30]]) / np.sqrt(2)

# Precompute extension and Fourier steps
extension = 4
wm = np.linspace(-NG * np.pi / L, NG * np.pi / L, extension * NG, endpoint=False)
wm1, wm2 = np.meshgrid(wm, wm)
s = np.sqrt(wm1 ** 2 + wm2 ** 2)
LT = 1.5 * L
green = (1 - sp.jv(0, LT * s)) / (s ** 2) - (LT * np.log(LT) * sp.jv(1, LT * s)) / s
green[extension * NG // 2, extension * NG // 2] = (LT ** 2 / 4 - LT ** 2 * np.log(LT) / 2)

J = np.fft.fftshift(wm) * np.ones([NG * extension, 1])
Kabsolute = np.transpose(np.sqrt(J ** 2 + np.transpose(J) ** 2))
Kabsolute[0, 0] = 1
Shat = (2 * sp.j1(r * Kabsolute) / (dx * Kabsolute)) ** 2
Shat[0, 0] = 1
Shat = Shat * (L / NG) ** 2 / (dx ** 2)
green1 = Shat * np.fft.fftshift(green)
green2 = Shat ** 2 * np.fft.fftshift(green)

# Precomputation for potential and acceleration fields
T1 = np.fft.ifftshift(np.fft.ifft2(green1))
T1 = T1[extension * NG // 4:extension * NG * 3 // 4, extension * NG // 4:extension * NG * 3 // 4]
T1 = np.fft.fft2(T1)

T2 = np.fft.ifftshift(np.fft.ifft2(green2))
T2 = T2[extension * NG // 4:extension * NG * 3 // 4, extension * NG // 4:extension * NG * 3 // 4]
T2 = np.fft.fft2(T2)

# Initialize particle positions and velocities
XP0 = np.random.randn(2, N) * sigmas
VP0 = np.random.randn(2, N)

energies = pd.DataFrame(columns=["DT", "Energy", "Type"])

# Loop over different time steps (DT)
for DT in [0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002, 0.0001]:
    # Initialize energy lists for kinetic and potential energy
    Eks = []  # Kinetic Energy
    Eps = []  # Potential Energy
    Etotals = []  # Total Energy
    Q = -1 / N
    XP = XP0
    VP = VP0
    NT = int(15.1 // DT)  # number of time steps

    for clock in range(NT):
        # Solve the electric field using Vico-Greengard and evaluate at particle positions using NUFFT
        raw = finufft.nufft2d1(XP[0, :] * np.pi / (L), XP[1, :] * np.pi / (L), 0j + np.ones(N), (2 * NG, 2 * NG), eps=1E-14,
                               modeord=1)

        rho_Hat = np.conjugate(raw) * Shat[::2, ::2] * Q
        phi_Hat = Q * T1 * np.conjugate(raw)
        coeff1 = Q * T2 * np.conjugate(raw) * -1j * np.transpose(J)[::2, ::2]
        coeff2 = Q * T2 * np.conjugate(raw) * -1j * J[::2, ::2]

        # Compute acceleration due to Electric Field
        a1 = np.array(np.real(
            finufft.nufft2d2(XP[0, :] * np.pi / (L) + np.pi, XP[1, :] * np.pi / (L) + np.pi, np.conjugate(coeff1),
                             eps=1e-14, modeord=1) * QM))
        a2 = np.array(np.real(
            finufft.nufft2d2(XP[0, :] * np.pi / (L) + np.pi, XP[1, :] * np.pi / (L) + np.pi, np.conjugate(coeff2),
                             eps=1e-14, modeord=1) * QM))
        a = np.array([a1, a2]) if N > 1 else np.array([[a1], [a2]])

        # Compute acceleration due to Magnetic Field using Boris Algorithm
        if not clock == 0:
            Vm = VP + a * DT / 2
            Vprime = Vm + np.cross(Vm, B0, axisa=0)[:, 0:2].T * QM * DT / 2
            Vp = Vm + np.cross(Vprime, B0, axisa=0)[:, 0:2].T * QM * DT / (1 + (np.linalg.norm(B0) * QM * DT / 2) ** 2)
            new_VP = Vp + a * DT / 2
            Ek = 0.5 * Q / QM * np.sum(((VP + new_VP) / 2) ** 2)
            VP = new_VP
        else:
            Ek = 0.5 * Q / QM * np.sum(VP ** 2)
            VP = VP + DT * (a + QM * np.cross(VP, B0, axisa=0)[:, 0:2].T) / 2
        XP = XP + DT * VP

        # Compute Energy
        Eks.append(Ek)
        rho = np.fft.fftshift(np.real(np.fft.ifft2(rho_Hat)))
        Ep = np.sum(np.fft.fft2(rho) * np.conjugate(phi_Hat) / (2 * L ** 2))
        Eps.append(Ep)
        Etotals.append(Ep + Ek)

        if clock % 2500 == 0:
            fourier_data = pd.DataFrame(np.fft.fftshift(np.abs(rho_Hat)), index=np.arange(-NG // 2, NG // 2, .5),
                                        columns=np.arange(-NG // 2, NG // 2, .5))
            rho_abs.append(fourier_data)
            rhos.append(-rho[int(0.5 * NG):int(1.5 * NG), int(.5 * NG):int(1.5 * NG)] / dx ** 2)

    energies = energies.append({"DT": DT, "Energy": np.max(np.abs(Etotals - Etotals[0])), "Type": "Numerical Error"},
                                   ignore_index=True)
    energies = energies.append({"DT": DT, "Energy": DT ** 2, "Type": "2nd Order Reference"}, ignore_index=True)

# Add additional code here for post-processing or plotting if needed.
