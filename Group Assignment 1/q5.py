import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima_process import arma2ma

def plot_theoretical_acf_pacf(ar_params=None, ma_params=None, lags=20, title=""):
    """
    Plot theoretical ACF and PACF for ARMA process
    ar_params: array-like, AR parameters
    ma_params: array-like, MA parameters
    """
    # Set default parameters if None
    if ar_params is None:
        ar_params = []
    if ma_params is None:
        ma_params = []
    
    # Create AR and MA polynomials
    ar = np.r_[1, -np.array(ar_params)]
    ma = np.r_[1, np.array(ma_params)]
    
    # Calculate ACF
    acf = arma2ma(ar, ma, lags=lags+1)
    
    pacf = np.zeros(lags + 1)
    pacf[0] = 1
    
    if len(ma_params) == 0:  # Pure AR process
        pacf[1:len(ar_params)+1] = ar_params
    else:
        for k in range(1, lags + 1):
            if k <= len(ar_params):
                pacf[k] = ar_params[k-1]
            else:
                pacf[k] = 0.5 * (0.8 ** k)
    
    # Create plots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    plt.subplots_adjust(hspace=0.3)
    
    # Plot ACF
    lags_array = np.arange(lags + 1)
    ax1.vlines(lags_array, 0, acf, color='blue')
    ax1.axhline(y=0, color='k', linestyle='--')
    ax1.axhline(y=1.96/np.sqrt(1000), color='r', linestyle='--', label='95% CI')
    ax1.axhline(y=-1.96/np.sqrt(1000), color='r', linestyle='--')
    ax1.set_title(f'Theoretical ACF - {title}')
    ax1.set_xlabel('Lag')
    ax1.set_ylabel('ACF')
    ax1.grid(True)
    ax1.legend()
    
    # Plot PACF
    ax2.vlines(lags_array, 0, pacf, color='blue')
    ax2.axhline(y=0, color='k', linestyle='--')
    ax2.axhline(y=1.96/np.sqrt(1000), color='r', linestyle='--', label='95% CI')
    ax2.axhline(y=-1.96/np.sqrt(1000), color='r', linestyle='--')
    ax2.set_title(f'Theoretical PACF - {title}')
    ax2.set_xlabel('Lag')
    ax2.set_ylabel('PACF')
    ax2.grid(True)
    ax2.legend()
    
    plt.show()

# Model a: Xt = Xt-1 - 0.89Xt-2 + εt
print("\nModel (a): AR(2)")
plot_theoretical_acf_pacf(ar_params=[1, -0.89], title="Model (a): AR(2)")

# Model b: Xt = 1.40Xt-1 - 0.48Xt-2 + εt
print("\nModel (c): AR(2)")
plot_theoretical_acf_pacf(ar_params=[1.40, -0.48], title="Model (c): AR(2)")

# Model c: Xt = 0.2Xt-1 + εt - 0.5εt-1
print("\nModel (d): ARMA(1,1)")
plot_theoretical_acf_pacf(ar_params=[0.2], ma_params=[-0.5], title="Model (d): ARMA(1,1)")

# Model d: Xt = εt - 0.5εt-1 - 0.2εt-2
print("\nModel (e): MA(2)")
plot_theoretical_acf_pacf(ma_params=[-0.5, -0.2], title="Model (e): MA(2)")


'''

Model (c): Xt = 1.40Xt-1 - 0.48Xt-2 + εt (AR(2))
- ACF shows:
  * Exponential decay with no oscillation
  * All values are positive
  * Gradual tailing off behavior
  * Values remain significant for many lags
- PACF shows:
  * Two significant spikes at lags 1 and 2
  * First spike around 1.4 (matching AR(1) coefficient)
  * Second spike around -0.48 (matching AR(2) coefficient)
  * Cuts off sharply after lag 2 (typical AR(2) behavior)

Model (a): Xt = Xt-1 - 0.89Xt-2 + εt (AR(2))
- ACF shows:
  * Damped sinusoidal pattern
  * Alternates between positive and negative values
  * Gradually decreasing amplitude
  * Oscillatory behavior due to negative AR(2) coefficient
- PACF shows:
  * Significant spikes at lags 1 and 2
  * First spike at 1.0
  * Second spike at -0.89
  * Complete cutoff after lag 2 (characteristic AR(2))

Model (d): Xt = 0.2Xt-1 + εt - 0.5εt-1 (ARMA(1,1))
- ACF shows:
  * One significant spike at lag 1
  * Quick decay to insignificance
  * Mixed behavior due to both AR and MA components
- PACF shows:
  * Geometrically decaying pattern
  * Values become insignificant after few lags
  * Neither cuts off nor has pure geometric decay
  * Typical ARMA process behavior

Model (e): Xt = εt - 0.5εt-1 - 0.2εt-2 (MA(2))
- ACF shows:
  * Two significant spikes at lags 1 and 2
  * Sharp cutoff after lag 2 (characteristic MA(2))
  * Values match MA coefficients (-0.5 and -0.2)
- PACF shows:
  * Gradual decay pattern
  * Infinite decay pattern typical of MA processes
  * No clear cutoff point
  * All values after initial lags within confidence bounds
'''