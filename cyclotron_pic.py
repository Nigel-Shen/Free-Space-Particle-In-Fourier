import numpy as np
import matplotlib.pyplot as plt
import scipy.special as sp
import matplotlib.animation as animation
from scipy import sparse
from tqdm import tqdm
import pandas as pd

fig, ax = plt.subplots()
artist = []
L = 1
NG = 32
DX = L / NG
QM = -1
N = 40000
WP = 1  # omega p
VT = 1  # Thermal Velocity
lambdaD = VT / WP  # Charge of a particle
# self.rho_back = - self.Q * self.N / self.L  # background rho
dx = L / NG
B0 = np.array([0,0,300])

sigmas = np.array([[1/10],[1/30]]) / np.sqrt(2)
# plt.scatter(XP[0,:], XP[1,:], alpha=0.1)
# plt.xlim([-0.5, 0.5])
# plt.ylim([-0.5, 0.5])
# plt.show()

# Prepare for convolution kernel:
extension = 4
wm = np.linspace(- NG * np.pi / L, NG * np.pi / L, extension*NG, endpoint=False) ## 8 times finer than regular Fourier step
wm1, wm2 = np.meshgrid(wm, wm)
s = np.sqrt(wm1**2 + wm2**2)
J = np.fft.fftshift(wm) * np.ones([NG * extension, 1])

## Construct mollified Green's function
LT = 1.5 * L ## Truncation window size
green = (1-sp.jv(0, LT*s)) / (s**2) - (LT*np.log(LT)*sp.jv(1, LT*s)) / s ## Green function in spectral space
green[extension*NG//2, extension*NG//2] = (LT**2/4 - LT**2*np.log(LT)/2)

## Precomputation
T1 = np.fft.ifftshift(np.fft.ifft2(np.fft.fftshift(green))) # * deltahat
T1 = T1[extension*NG//4:extension*NG*3//4, extension*NG//4:extension*NG*3//4]
T1 = np.fft.fft2(T1) # This is the kernel for potential field

rhos = []

energies = pd.DataFrame(columns=["Method", "Time", "Energy"])
XP0 = np.random.randn(2, N) * sigmas + np.array([[0.5], [0.5]])
VP0 = np.random.randn(2, N)
for DT in [0.0005]: #[0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002, 0.0001]:
    Eks = [] # Kinetic Energy
    Eps = [] # Potential Energy
    Etotals = [] # Total Energy
    Q = - 1 / N
    XP = XP0
    VP = VP0
    NT = int(0.2 // DT)  # number of time steps
    for clock in tqdm(range(NT)):
        '''
        1. Solve the electric field using Vico-Greengard and evaluate at particle positions using NUFFT
        
        2. Push the particles using leapfrog
        '''
        g0, g1 = np.floor(XP[0] / DX).astype(int), np.floor(XP[1] / DX).astype(int)  # which grid point to project onto
        g = np.array([[g0 - 1, g0, g0 + 1],[g1 - 1, g1, g1 + 1]])  # used to determine bc
        a, b = XP[0] % DX, XP[1] % DX
        c1, c2, c3, c4 = (DX-a)**2, (DX-b)**2, DX**2 + 2 * DX * a - 2 * a**2, DX**2 + 2 * DX * b - 2 * b**2
        tot = (DX * DX) ** 2
        A = c1 * c2 / (4*tot)
        B = c2 * c3 / (4*tot)
        C = a**2 * c2/ (4*tot)
        D = c1 * c4 / (4*tot)
        F = a**2 * c4 / (4*tot)
        G = b**2 * c1 / (4*tot)
        H = b**2 * c3 / (4*tot)
        I = a**2 * b**2 / (4*tot)
        E = 1 - A - B - C - D - F - G - H - I
        fraz = np.array([A, B, C, D, E, F, G, H, I])
        # apply bc on the projection
        matrices = []
        for i in range(3):
            for j in range(3):
                matrices.append(sparse.csr_matrix((fraz[3*i+j], (np.linspace(0, N - 1, N).astype(int), int(L/DX) * g[0,i] + g[1,j])), shape=(N, int(L/DX)**2)))
        M = sum(matrices)
        rho = (Q / (DX*DX)) * M.sum(0).reshape([int(L/DX), int(L/DX)])
        phi_Hat = T1*np.fft.fft2(rho, s=[extension*NG//2, extension*NG//2]) * 4
        E1_Hat = phi_Hat * -1j * np.transpose(J)[::2, ::2]
        E2_Hat = phi_Hat * -1j * J[::2, ::2]
        E1 = np.fft.ifft2(E1_Hat)[extension*NG//4:extension*NG//2, extension*NG//4:extension*NG//2]
        E2 = np.fft.ifft2(E2_Hat)[extension*NG//4:extension*NG//2, extension*NG//4:extension*NG//2]
        a1 = np.transpose(M * np.real(E1).flatten()) * QM
        a2 = np.transpose(M * np.real(E2).flatten()) * QM

        a = np.array([a1, a2])
        # Compute Acceleration due to Magnetic Field using Boris Algorithm
        if not clock==0:
            Vm = VP + a * DT / 2
            Vprime = Vm + np.cross(Vm, B0, axisa=0)[:, 0:2].T * QM * DT / 2
            Vp = Vm + np.cross(Vprime, B0, axisa=0)[:, 0:2].T * QM * DT / (1 + (np.linalg.norm(B0)*QM*DT/2) ** 2)
            new_VP = Vp + a * DT / 2
            # new_VP1 = (QM * B0[2]*DT/2 *(VP[1,:]+DT*(a2-QM*B0[2]*VP[0,:]/2)) + VP[0,:] + DT * (a1+QM*B0[2]*VP[1,:]/2)) / (1+(QM*B0[2]*DT/2)**2)
            # new_VP2 = (-QM * B0[2]*DT/2 *(VP[0,:]+DT*(a1+QM*B0[2]*VP[1,:]/2)) + VP[1,:] + DT * (a2-QM*B0[2]*VP[0,:]/2)) / (1+(QM*B0[2]*DT/2)**2)
            Ek = 0.5 * Q / QM * np.sum(((VP + new_VP) / 2) ** 2)
            VP = new_VP
        else:
            Ek = 0.5 * Q / QM * np.sum(VP ** 2)
            VP = VP + DT * (a + QM * np.cross(VP, B0, axisa=0)[:, 0:2].T) / 2
        XP = XP + DT * VP

        if clock%4000==0:
            print(clock)
            # container = ax.imshow(-rho)#[int(NG):int(2*NG), int(NG):int(2*NG)])
            # fig.colorbar(container)
            # artist.append([container])
            rhos.append(-rho)
        ## Compute Energy
        Eks.append(Ek)
        rho = np.fft.fftshift(rho)
        Ep = np.sum(rho * np.fft.ifft2(phi_Hat)[extension*NG//4:extension*NG//2, extension*NG//4:extension*NG//2] * DX ** 2 / 2)
        Eps.append(Ep)
        Etotals.append(Ep + Ek)
        energies = energies.append({"Method": "FSPIC", "Time": clock * DT, "Energy": (Ep + Ek) / (Eps[0] + Eks[0])}, ignore_index=True)
    #plt.clf()
    #tick = np.linspace(0, clock*DT, clock+1, endpoint=False)
    #plt.plot(tick, (np.array(Etotals) - Etotals[0]) / Etotals[0], linewidth='2', label=r"$DT = $" + str(DT))
    #if DT < 0.001:
    #    plt.plot(tick, (np.array(Etotals) - Etotals[0]) / Etotals[0] * (0.001 / DT) ** 2, ls='--', linewidth='1.5', label= str((0.001 / DT) ** 2) + r"$\times DT = $" + str(DT))
    #dE.append(np.max(np.abs(np.array(Etotals) - Etotals[0])))
#plt.legend()
#plt.xlabel('t')
#plt.ylabel('(E-E0)/E0')
#plt.show()
# ani = animation.ArtistAnimation(fig=fig, artists=artist, interval=40)
# ani.save(filename="pic_cyclotron.gif", writer="Pillow")
#tick = np.linspace(0, clock*DT, clock+1, endpoint=False)
#plt.clf()
#plt.plot(tick, Etotals, label='Total Energy')
#plt.plot(tick, Eks, label='Kinetic Energy')
#plt.plot(tick, Eps, label='Potential Energy')
#plt.xlim(0, 0.5)
#plt.loglog([0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002, 0.0001], dE, marker='o', label='Energy Total Variation')
#plt.loglog([0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002, 0.0001], np.array([0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002, 0.0001]) ** 2, label='2nd Order Reference')
#plt.legend()
# plt.scatter(XP[0,0], XP[1,0])
# plt.scatter(XP[0,1], XP[1,1])
# plt.xlim(-0.5, 0.5)
# plt.ylim(-0.5, 0.5)
#plt.xlabel(r"$\Delta t$")
#plt.ylabel(r"$||E-E(0)||$")
#plt.show()