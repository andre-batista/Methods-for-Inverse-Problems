# Load libraries
import numpy as np
import matplotlib.pyplot as plt

# Discretization size
n = 16

# Tikhonov parameter
ALPHA = np.array([1e-1,1e-2,1e-3,1e-4,1e-5,1e-6,1e-7,1e-8,1e-9,1e-10])

# Noise size
delta = np.array([.0001,.001,.01,.1])

# Error array
error = np.zeros((delta.size,ALPHA.size))

for i in range(delta.size):
    t = np.linspace(0,1,n+1) # Time array
    y = np.exp(t) # Data array
    x = np.ones(y.size) # Exact solution array
    xa = np.zeros((n+1,ALPHA.size)) # Approximation array

    # Add noise
    yd = np.zeros(y.size)
    for k in range(y.size):
        yd[k] = np.random.normal(loc=y[k],scale=.9*delta[i])

    # Simpson rule
    A = np.zeros((t.size,t.size))
    for k in range(t.size):
        A[k,0] = 1/3/n*(1+t[k]*t[0])*np.exp(t[k]*t[0]) # j = 0
        A[k,-1] = 1/3/n*(1+t[k]*t[-1])*np.exp(t[k]*t[-1]) # j = n
        j_odd = np.arange(1,t.size-1,2) 
        j_even = np.arange(2,t.size-1,2)
        A[k,j_odd] = 4/3/n*(1+t[k]*t[j_odd])*np.exp(t[k]*t[j_odd]) # j = 1,3,...,n-1
        A[k,j_even] = 2/3/n*(1+t[k]*t[j_even])*np.exp(t[k]*t[j_even]) # j = 2,4,...,n-2

    for j in range(ALPHA.size):
        alpha = ALPHA[j]
        # Regularization function
        R = np.dot(np.linalg.inv(alpha*np.identity(t.size)+np.linalg.matrix_power(A,2)),A)
        xa[:,j] = np.dot(R,yd) # Approximated solution
        error[i,j] = np.sqrt(1/(n+1)*np.sum((x-xa[:,j])**2)) # Error measure

    # Plot solutions
    fig = plt.gca()
    fig.plot(t,x,label='Exact')
    plt.legend(r'$\alpha$')
    for j in range(ALPHA.size):
        fig.plot(t,xa[:,j],'--*',label=r'$\alpha = $%.1e' %ALPHA[j])
    fig.legend()
    plt.xlabel(r'$t$')
    plt.ylabel(r'$x(t)$')
    plt.title(r'$\delta =$ %.0e' %delta[i])
    plt.grid()
    plt.show()

# Print table
print('--------------------------------------------------------')
title = 'alpha     '
for i in range(delta.size):
    title = title + 'd = %.0e  ' %delta[i]
print(title)
print('--------------------------------------------------------')
for j in range(ALPHA.size):
    row = '%.1e   ' %ALPHA[j]
    for i in range(delta.size):
        row = row + '%.1e    ' %error[i,j]
    print(row)
print('--------------------------------------------------------')