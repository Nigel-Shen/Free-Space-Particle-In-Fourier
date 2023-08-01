# Free Space Particle-In-Fourier

This repository contains Python code for simulating a cyclotron beam in free space using a modified Particle-In-Cell (PIC) method, which we call Free Space Particle-In-Fourier (FSPIF). It inherits the good energy conservation in PIF algorithm, while solves for free space BCs with high accuracy.

## Description

The code simulates the behavior of particles in a cyclotron beam, considering their interactions with electric and magnetic fields. It uses the Particle-In-Cell technique to handle the dynamics of particles and numerical methods to compute electric and magnetic field interactions.

## Getting Started

### Prerequisites

- Python 3.x
- NumPy
- SciPy
- matplotlib
- finufft
- pandas

### Installation

1. Clone the repository:

```
git clone https://github.com/Nigel-Shen/Free-Space-Particle-In-Fourier.git
```

2. Install the required Python packages:

```
pip install numpy scipy matplotlib finufft pandas
```

### Usage

Run the `free_space_cyclotron.py` script to perform the simulation with the default settings. You can adjust parameters like time step (DT) and number of particles (N) inside the script.

```
python free_sapce_cyclotron.py
```

### Parameters

- `L`: Length of the simulation domain.
- `NG`: Number of grid points used for the Fourier transform of the electric field.
- `N`: Number of particles in the cyclotron beam.
- `QM`: Charge-to-mass ratio of the particles.
- `WP`: Plasma frequency of the particles.
- `VT`: Thermal velocity of the particles.
- `lambdaD`: Debye length of the plasma.
- `dx`: Grid spacing in the simulation domain.
- `B0`: Magnetic field vector at the simulation domain.
- `sigmas`: Standard deviations of the initial particle positions and velocities.
- `DT`: Time step used for the simulation.
- `extension`: Factor to extend the Fourier steps for the Green's function convolution kernel.
- `NT`: Number of time steps for the simulation.

### Results

The simulation will generate various plots and animations representing the behavior of the cyclotron beam over time. Energy variations and other relevant data will be stored in a Pandas DataFrame for analysis.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- The code is based on scientific research by Nigel Shen, Antoine Cerfon and Sriramkrishnan Muralikrishnan.

