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

rho_abs = [] # Stores absolute value of /rho in Fourier space
rhos = [] # Stores /rho in physical space

L = 1 # Side length of domain
NG = 32 # Resolution
QM = -1 # Q/M
N = 40000 # Number of particles
WP = 1  # omega p
VT = 1  # Thermal Velocity
lambdaD = VT / WP # Debye length
dx = L / NG # Side length of a single sell
B0 = np.array([0,0,300]) # Constant magnetic field
sigmas = np.array([[1/10],[1/30]]) / np.sqrt(2) # Control the shape of the beam

# Prepare for convolution kernel:
extension = 4
wm = np.linspace(- NG * np.pi / L, NG * np.pi / L, extension*NG, endpoint=False) ## 4 times finer than regular Fourier step
wm1, wm2 = np.meshgrid(wm, wm)
s = np.sqrt(wm1**2 + wm2**2)

## Construct mollified Green's function
LT = 1.5 * L ## Truncation window size
green = (1-sp.jv(0, LT*s)) / (s**2) - (LT*np.log(LT)*sp.jv(1, LT*s)) / s ## Green function in spectral space
green[extension*NG//2, extension*NG//2] = (LT**2/4 - LT**2*np.log(LT)/2)

r = L / NG
J = np.fft.fftshift(wm) * np.ones([NG * extension, 1])
Kabsolute = np.transpose(np.sqrt(J**2 + np.transpose(J)**2))
Kabsolute[0,0] = 1  # avoid 0 on denominator
Shat = (2 * sp.j1(r * Kabsolute) / (r * Kabsolute)) ** 2 
Shat[0, 0] = 1
Shat = Shat * (L / NG) ** 2 / (r **2)

green1 = Shat * np.fft.fftshift(green)
green2 = Shat ** 2 * np.fft.fftshift(green)

## Precomputation
'''
For optimal performance use precomputation; for optimal accuracy do not use precomputation.
'''
T1 = np.fft.ifftshift(np.fft.ifft2(green1)) 
T1 = T1[extension*NG//4:extension*NG*3//4, extension*NG//4:extension*NG*3//4]
T1 = np.fft.fft2(T1) # This is the kernel for potential field

T2 = np.fft.ifftshift(np.fft.ifft2(green2))
T2 = T2[extension*NG//4:extension*NG*3//4, extension*NG//4:extension*NG*3//4]
T2 = np.fft.fft2(T2) # This is the kernel for acceleration

## Initialize particle positions and velocities
XP0 = np.random.randn(2, N) * sigmas
VP0 = np.random.randn(2, N)

## Create dataframe needed for post-processing
# energies = pd.DataFrame(columns=["Method", "Time", "Energy"]) # Compare with PIC
# energies = pd.DataFrame(columns=["DT", "Energy", "Type"]) # Show global convergence
energies = pd.DataFrame(columns=["DT", "Energy", "Time"]) # Show local convergence

## Main simulation cycles
for DT in [0.001, 0.0005, 0.00025]: #[0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002, 0.0001]:
    Eks = [] # Kinetic energy
    Eps = [] # Potential energy
    Etotals = [] # Total energy
    Q = - 1 / N # Total charge
    XP = XP0
    VP = VP0
    NT = int(0.2 // DT)  # Number of time steps
    for clock in range(NT):
        '''
        1. Solve the electric field using Vico-Greengard and evaluate at particle positions using NUFFT
        
        2. Push the particles using Boris algorithm
        '''

        # NUFFT Type 1 to evaluate exp(-ikX)
        raw = finufft.nufft2d1(XP[0, :] * np.pi / (L), XP[1, :] * np.pi / (L), 0j + np.ones(N), (2*NG, 2*NG), eps=1E-14, modeord=1)
        
        # Compute Electric Field
        rho_Hat = np.conjugate(raw) * Shat[::2,::2] * Q # /rho in Fourier space
        phi_Hat = Q * T1 * np.conjugate(raw) # /phi in Fourier space
        coeff1 = Q * T2 * np.conjugate(raw) * -1j * np.transpose(J)[::2, ::2] # Not exactly Electric field! Notice it convolutes twice with shape function
        coeff2 = Q * T2 * np.conjugate(raw) * -1j * J[::2, ::2] 

        # Compute Acceleration due to Electric Field
        a1 = np.array(np.real(finufft.nufft2d2(XP[0, :] * np.pi / (L) + np.pi, XP[1, :] * np.pi / (L) + np.pi, np.conjugate(coeff1), eps=1e-14, modeord=1) * QM))
        a2 = np.array(np.real(finufft.nufft2d2(XP[0, :] * np.pi / (L) + np.pi, XP[1, :] * np.pi / (L) + np.pi, np.conjugate(coeff2), eps=1e-14, modeord=1) * QM))
        
        if N==1:
            a = np.array([[a1], [a2]])
        else:
            a = np.array([a1, a2])

        # Compute Acceleration due to Magnetic Field using Boris Algorithm
        if not clock==0:
            Vm = VP + a * DT / 2
            Vprime = Vm + np.cross(Vm, B0, axisa=0)[:, 0:2].T * QM * DT / 2
            Vp = Vm + np.cross(Vprime, B0, axisa=0)[:, 0:2].T * QM * DT / (1 + (np.linalg.norm(B0)*QM*DT/2) ** 2)
            new_VP = Vp + a * DT / 2
            Ek = 0.5 * Q / QM * np.sum(((VP + new_VP) / 2) ** 2)
            VP = new_VP
        else:
            Ek = 0.5 * Q / QM * np.sum(VP ** 2)
            VP = VP + DT * (a + QM * np.cross(VP, B0, axisa=0)[:, 0:2].T) / 2
        XP = XP + DT * VP

        ## Compute Energy
        Eks.append(Ek)
        rho = np.fft.fftshift(np.real(np.fft.ifft2(rho_Hat)))
        Ep = np.sum(np.fft.fft2(rho) * np.conjugate(phi_Hat) / (2 * L ** 2))
        Eps.append(Ep)
        Etotals.append(Ep+Ek)
        # energies = energies.append({"Method": "FSPIF", "Time": clock * DT, "Energy": np.real((Ep + Ek) / (Eps[0] + Eks[0]))}, ignore_index=True)
        energies = energies.append({"DT": DT, "Time": clock * DT, "Energy": np.real((Ep + Ek) / (Eps[0] + Eks[0]))}, ignore_index=True)
        
        ## Time slices
        if clock%2500==0:
            print(clock)
            fourier_data = pd.DataFrame(np.fft.fftshift(np.abs(rho_Hat)), index=np.arange(-NG//2, NG//2, .5), columns=np.arange(-NG//2, NG//2, .5))
            rho_abs.append(fourier_data)
            rhos.append(-rho[int(0.5*NG):int(1.5*NG), int(.5*NG):int(1.5*NG)] / dx ** 2)
            
    # energies = energies.append({"DT": DT, "Energy": np.max(np.abs(Etotals-Etotals[0])), "Type": "Numerical Error"}, ignore_index=True)
    # energies = energies.append({"DT": DT, "Energy": DT ** 2, "Type": "2nd Order Reference"}, ignore_index=True)

# Add additional code here for post-processing or plotting if needed.
