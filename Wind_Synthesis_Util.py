"""

Synthesizes wind noise using the von Kármán wind turbulence model
Approximate using only the longitudinal component of wind (low turbulence) 

https://en.wikipedia.org/wiki/Von_K%C3%A1rm%C3%A1n_wind_turbulence_model


"""

from matplotlib import pyplot as plt
import numpy as np


def addWindNoise(base: np.ndarray, fs: float, f_min: float = 0.1, f_max: float = 10, V: float = 5, sigma: float = 1, L: float = 50) -> np.ndarray[np.float64]:
    '''
    Adds synthetic wind noise to provided base signal.

    Args:
        base (np.ndarray): Base signal
        fs (float): Sampling frequency, Hz
        f_min (float, optional): Lower bound of frequencies to generate, Hz. Defaults to 0.1 Hz.
        f_max (float, optional): Upper bound of frequencies to generate, Hz. Defaults to 10 Hz.
        V (float, optional): Mean wind speed, m/s. Defaults to 5 m/s.
        sigma (float, optional): Variance of wind speed, m/s squared. Defaults to 1 (m/s)/s.
        L (float, optional): Scale length (size of eddies), m. Defaults to 50 m.

    Returns:
        np.ndarray: Base signal with semi-random wind noise added.
    '''
    N = len(base) 

    # Generate frequency bins
    f = np.fft.rfftfreq(N, 1/fs)

    # Convert Hz to rad/s
    omega = 2 * np.pi * f / V # 

    # von Kármán model
    S = V * (sigma**2) * (2 * L / np.pi) * (1.0 + (1.339 * L * omega) ** 2) ** (-5.0/6.0)
    
    # Zero out frequencies outside range
    S[f < f_min] = 0.0
    S[f > f_max] = 0.0

    # Convert power to amplitude
    df = f[1] - f[0]
    A = np.sqrt(S * df)

    # Random phase angles for each bin
    random_phases = np.exp(1j * (2 * np.pi * np.random.rand(len(f))))

    # Get spectrum by multiplying amplitude and phase
    spectrum = A * random_phases

    # Convert back into time domain
    noise = np.fft.irfft(spectrum, n=N)

    # Convert m/s (from input units) into mB
    p_mb = 1.225 * V * noise * (1/100)

    # Plot
    t = np.arange(N) / fs
    plt.figure(figsize=(10,4))
    plt.plot(t, p_mb)
    plt.xlabel("Time [s]")
    plt.ylabel("Pressure [mB]")
    plt.title("Wind noise in millibars")
    plt.show()

    # Add to base
    return base + p_mb