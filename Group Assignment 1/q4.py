import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf, pacf

np.random.seed(42)

# Generate time series
t = np.arange(-49, 51)
U = np.random.uniform(0, 1)
X = np.cos(2 * np.pi * (t/12 + U))

# Calculate ACF and PACF
nlags = 30
acf_values = acf(X, nlags=nlags)
pacf_values = pacf(X, nlags=nlags)

# Create confidence intervals for ACF and PACF
conf_int = 1.96/np.sqrt(len(X))  # 95% confidence interval

# Create subplot
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))
plt.subplots_adjust(hspace=0.3)

# Plot Time Series
ax1.plot(t, X)
ax1.set_title('Time Series Plot')
ax1.set_xlabel('Time')
ax1.set_ylabel('Value')
ax1.grid(True)

# Plot ACF
lags = np.arange(nlags + 1)
ax2.vlines(lags, [0], acf_values, color='blue')
ax2.plot(lags, [conf_int]*len(lags), 'r--', label='95% Confidence Interval')
ax2.plot(lags, [-conf_int]*len(lags), 'r--')
ax2.set_title('Autocorrelation Function (ACF)')
ax2.set_xlabel('Lag')
ax2.set_ylabel('ACF')
ax2.grid(True)
ax2.legend()

# Plot PACF
ax3.vlines(lags, [0], pacf_values, color='blue')
ax3.plot(lags, [conf_int]*len(lags), 'r--', label='95% Confidence Interval')
ax3.plot(lags, [-conf_int]*len(lags), 'r--')
ax3.set_title('Partial Autocorrelation Function (PACF)')
ax3.set_xlabel('Lag')
ax3.set_ylabel('PACF')
ax3.grid(True)
ax3.legend()

# Print some key statistics
print(f"Random U value: {U:.4f}")
print(f"\nFirst few observations of the time series:")
print(X[:5])
print(f"\nACF values for first 5 lags:")
print(acf_values[:5])
print(f"\nPACF values for first 5 lags:")
print(pacf_values[:5])

plt.show()

'''
Key Observations:
1. The series is clearly periodic with a cycle length of 12 time units
2. The ACF shows strong periodic correlation, confirming the cyclical nature
3. The PACF cuts off after the first few lags, suggesting that most of the correlation structure can be captured by these initial lags
4. The series is stationary (constant mean and variance over time)

This analysis reveals that the time series has a strong periodic component, which is expected given that it's generated from a cosine function.
The period of 12 units is clearly visible in both the original series and its autocorrelation structure.

'''