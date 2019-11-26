# Import libraries
import numpy as np
import matplotlib.pyplot as plt

# Main parameters
n = 32 # Discretization parameters
delta = 1e-2 # noise 
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

# Add noise to data
if delta == 0:
    gd = np.copy(g)
else:
    gd = np.zeros(g.size)
    for k in range(g.size):
        gd[k] = np.random.normal(loc=g[k],scale=.9*delta)

# Base function evaluation
x = np.zeros((t.size,2*n))
for i in range(t.size):
    for j in range(2*n):
        if j is 0:
            if t[i] < np.pi/2/n or t[i] > 2*np.pi-np.pi/2/n:
                x[i,j] = np.sqrt(n/np.pi)
            else:
                x[i,j] = .0
        else:
            if np.abs(t[i]-t[j]) < np.pi/2/n:
                x[i,j] = np.sqrt(n/np.pi)
            else:
                x[i,j] = .0

# Coefficient matrix for linear system
A = np.zeros((N,N),dtype=complex)
for k in range(N):
    for j in range(N):
        A[k,j] = K[k,:]@x[:,j]

# Beta
beta = np.copy(gd)

# Solve A*alpha=beta
alpha = np.linalg.solve(A,beta)

# Recover psi function
psi_d = np.zeros(t.size)
j = np.arange(-n,n)
for i in range(t.size):
    psi_d[i] = np.sum(alpha*x[i,:])

# Compute error ||psi-psi_d||_L^2
error = np.sqrt(np.sum(np.abs(psi-psi_d)**2)*(t[1]-t[0]))
error_per = np.mean(np.abs((psi-psi_d)/psi))

# Print model and results
print('COLLOCATION OF SYMM\'S EQUATION')
print('n = %d' %n)
print('delta = %.2e' %delta)
print('error = %.2e' %error)
print('Average = %.2f %%' %error_per)

# Plot results
plt.plot(t,psi)
plt.plot(t,psi_d,'--*')
plt.xlabel(r'$t$')
plt.ylabel(r'$\psi(t)$')
plt.title('Colloction of Symm\'s Equation')
plt.legend(('Exact',r'$\psi^\delta$'))
plt.grid()
plt.show()