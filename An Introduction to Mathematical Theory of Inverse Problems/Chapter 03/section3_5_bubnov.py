# Import libraries
import numpy as np
import matplotlib.pyplot as plt

# General parameters
N = np.array([1,2,3,4,5,6,10,12,15,20,30,40,50,60,80,100]) # discretization parameter
DEL = np.array([1e-1,1e-2,1e-3,0]) # noise size
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

# Allocate error table
table = np.zeros((N.size,DEL.size))

for idx_n in range(N.size):
    n = N[idx_n]

    # Time array
    t = np.arange(0,2*n)*np.pi/n
    Nt = t.size # array size

    # Solution array
    psi = psifun(t)

    # Operator matrix
    K = np.zeros((Nt,Nt))
    for k in range(Nt):
        for j in range(Nt):
            K[k,j] = Rl(np.abs(k-j),n) + np.pi/n*kfun(t[k],t[j])

    # Data array
    g = K@psi

    for idx_d in range(DEL.size):
        delta = DEL[idx_d]

        # Add noise to data
        if delta == 0:
            gd = np.copy(g)
        else:
            gd = np.zeros(g.size)
            for k in range(g.size):
                gd[k] = np.random.normal(loc=g[k],scale=.9*delta)

        # Coefficient matrix for Galerkin Least Squares Method
        A = np.zeros((Nt,Nt),dtype=complex)
        for i in range(Nt):
            for j in range(Nt):
                A[i,j] = np.vdot(K@psi_h(i,t),psi_h(j,t))

        # Compute beta 
        beta = np.zeros(Nt,dtype=complex)
        for i in range(Nt):
            beta[i] = np.vdot(psi_h(i,t),gd)

        # Compute alpha
        alpha = np.linalg.solve(A,beta)

        # Compute psi_ad
        psi_ad = np.zeros(Nt)
        for i in range(Nt):
            j = np.arange(Nt)
            psi_ad[i] = np.real(np.sum(alpha*psi_h(j,t[i])))

        # Compute error
        table[idx_n,idx_d] = np.sqrt(1/(Nt+1)*np.sum((psi-psi_ad)**2))

        # Plot solution
        if False:
            plt.plot(t,psi)
            plt.plot(t,psi_ad,'--*')
            plt.xlabel(r'$t$')
            plt.ylabel(r'$\psi(t)$')
            plt.title(r'Bubnov-Galerkin Method - $n =$ %d' %n 
                    + r' - $\delta = $%.0e' %delta)
            plt.legend(('Exact',r'$\psi^{\alpha,\delta}$'))
            plt.grid()
            plt.show()

# Print experiment information
line = '----'
for delta in DEL:
    line = line + '-----------'
print(line)
print('BUBNOV-GALERKIN METHOD')
print(line)
title = ' m   '
for delta in DEL:
    title = title + 'd = %.0e  ' %delta
print(title)
print(line)
for idx_n in range(N.size):
    row = '%3d ' %N[idx_n]
    for idx_d in range(DEL.size):
        row = row + ' %.2e  ' %table[idx_n,idx_d]
    print(row)
print(line)

# Plot convergence
nrow = int(np.sqrt(DEL.size))
if np.remainder(DEL.size,nrow) == 0:
    ncol = nrow
else:
    ncol = nrow+1
idx_fig = 1
for idx_d in range(DEL.size):
    plt.subplot(nrow*100+ncol*10+idx_fig)
    plt.semilogy(N,table[:,idx_d],'--*')
    plt.xlabel(r'$n$')
    plt.ylabel(r'$|\psi^{\alpha,\delta}-\psi|_2$')
    plt.title(r'Bubnov-Galerkin Method - $n =$ %d' %n)
    plt.grid()
    idx_fig+=1
plt.show()
    