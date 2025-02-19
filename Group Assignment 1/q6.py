import numpy as np
from scipy import stats

def generate_ar1_series(n, phi, sigma_z=1):
    """Generate AR(1) series: Xt = phi*Xt-1 + Zt"""
    np.random.seed(42)
    
    # Generate initial value X0 ~ Normal(0, 1/(1-phi^2))
    x0 = np.random.normal(0, 1/np.sqrt(0.36))
    
    # Generate the series
    x = np.zeros(n)
    x[0] = x0
    z = np.random.normal(0, sigma_z, n)
    
    for t in range(1, n):
        x[t] = phi * x[t-1] + z[t]
    
    return x

def generate_ma1_series(n, theta, sigma_z=1):
    """Generate MA(1) series: Xt = theta*Zt-1 + Zt"""
    np.random.seed(42)

    z = np.random.normal(0, sigma_z, n+1)
    
    # Generate X series
    x = np.zeros(n)
    for t in range(n):
        x[t] = theta * z[t] + z[t+1]
    
    return x

def compute_autocorr_matrix(x, order=4):
    """Compute autocorrelation matrix"""
    n = len(x)
    
    # Calculate autocorrelations up to order
    acf = np.zeros(order)
    x_mean = np.mean(x)
    x_var = np.var(x)
    
    for k in range(order):
        c = 0
        for t in range(k, n):
            c += (x[t] - x_mean) * (x[t-k] - x_mean)
        acf[k] = c / ((n-k) * x_var)
    
    # Construct autocorrelation matrix
    R = np.zeros((order, order))
    for i in range(order):
        for j in range(order):
            if i == j:
                R[i,j] = 1
            else:
                R[i,j] = acf[abs(i-j)]
    
    return R, acf

n = 100

# Model (a): AR(1)
x_ar = generate_ar1_series(n, phi=0.8)
R_ar, acf_ar = compute_autocorr_matrix(x_ar)

# Model (b): MA(1)
x_ma = generate_ma1_series(n, theta=0.8)
R_ma, acf_ma = compute_autocorr_matrix(x_ma)

print("Model (a): AR(1) Autocorrelation Matrix")
print(R_ar)
print("\nFirst 4 autocorrelations:", acf_ar)

print("\nModel (b): MA(1) Autocorrelation Matrix")
print(R_ma)
print("\nFirst 4 autocorrelations:", acf_ma)