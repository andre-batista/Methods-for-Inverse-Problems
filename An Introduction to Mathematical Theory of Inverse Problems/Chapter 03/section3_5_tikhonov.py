# Import libraries
import numpy as np
import matplotlib.pyplot as plt

# Main parameters
n = 60 # Discretization parameters
delta = np.array([1e-1,1e-2,1e-3,0]) # noise
Na = 40 # Number of values for Tikhonov parameter
def psifun(t):  # Solution function for Kx=y
    return np.exp(3*np.sin(t))

# Auxiliar functions
def gamma(s):
    return np.cos(s) + 1j*2*np.sin(s)
def gamma_p(s):
    return -np.sin(s) + 1j*2*np.cos(s)
def Rl(l,n):
    m = np.arange(1,n)
    return 1/n*((-1)**l/2/n+np.sum(1/m*np.cos(m*l*np.pi/n)))
def kfun(t,s):
    if t != s:
        return -1/2/np.pi*np.log(np.abs(gamma(t)-gamma(s))**2/4/np.sin((t-s)/2)**2)
    else:
        return -1/np.pi*np.log(np.abs(gamma_p(t)))
def psi_h(j,t):
    return np.exp(1j*j*t)
def geterror(x):
    return 1/x.size*np.sum(np.abs(x)**2)

t = np.arange(2*n)*np.pi/n # Time array
N = t.size # Number of points of integration

# Solution array
psi = psifun(t)

# Operator matrix
K = np.zeros((N,N))
for k in range(N):
    for j in range(N):
        K[k,j] = Rl(np.abs(k-j),n) + np.pi/n*kfun(t[k],t[j])

# Data array
g = K@psi

# Allocate error arrays
error_psi = np.zeros((Na,delta.size))
error_g = np.zeros((Na,delta.size))

for idx_d in range(delta.size):

    # Tikhonov parameter
    if delta[idx_d] != 0:
        alpha = np.linspace(.05*delta[idx_d],delta[idx_d],Na) 
    else:
        alpha = np.linspace(1e-5,1e-3,Na)

    # Add noise to data
    if delta[idx_d] == 0:
        gd = np.copy(g)
    else:
        gd = np.zeros(g.size)
        for k in range(g.size):
            gd[k] = np.random.normal(loc=g[k],scale=1.5*delta[idx_d])

    for idx_a in range(Na):

        # Compute regularization operator
        Ra = np.linalg.inv(alpha[idx_a]*np.eye(N)+K@K)@K

        # Solve equation
        psi_ad = Ra@gd

        # Compute error
        error_psi[idx_a,idx_d] = geterror(psi_ad-psi)
        error_g[idx_a,idx_d] = geterror(K@psi_ad-gd)

    # Plot results
    plt.plot(alpha,error_psi[:,idx_d])
    plt.plot(alpha,error_g[:,idx_d])
    plt.xlabel(r'$\alpha$')
    plt.ylabel('Error')
    plt.title(r"Symm's Equation - Tikhonov Method - $\delta =$ %.0e" %delta[idx_d])
    plt.legend((r'$|\tilde{\psi}^{\alpha,\delta}-\tilde{\psi}|_2$',
                r'$|A\tilde{\psi}^{\alpha,\delta}-\tilde{g}|_2$'))
    plt.grid()
    plt.show()

print(np.min(error_psi[:,0])/np.min(error_psi[:,1]))