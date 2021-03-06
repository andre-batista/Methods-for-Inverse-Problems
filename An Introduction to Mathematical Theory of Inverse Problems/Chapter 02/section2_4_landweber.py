# Load libraries
import numpy as np
import matplotlib.pyplot as plt


n = 16 # Discretization size
a = .5 # Landweber parameter
M = np.array([1,2,3,4,5, # Iterations
              6,7,8,9,10,
              100,200,300,
              400,500])
delta = np.array([.0001,.001,.01,.1]) # Noise size

# Error array
table = np.zeros((delta.size,M.size))
convergence = np.zeros((np.amax(M),delta.size))

for i in range(delta.size):
    t = np.linspace(0,1,n+1) # Time array
    y = np.exp(t) # Data array
    x = np.ones(y.size) # Exact solution array
    xa = np.zeros(x.shape) # Approximation array
    
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

    # Plot solutions
    fig = plt.gca()
    fig.plot(t,x,label='Exact')
    plt.legend(r'$\alpha$')

    # Landweber iteration
    xa = np.zeros(x.size)
    R = np.identity(t.size)-a*np.linalg.matrix_power(A,2)
    S = a*np.dot(A,yd)
    k = 0
    for m in range(1,np.max(M)+1):
        xa = np.dot(R,xa) + S
        convergence[m-1,i] = np.sqrt(1/(n+1)*np.sum((x-xa)**2))
        if m == M[k]:
            table[i,k] = convergence[m-1,i]
            fig.plot(t,xa,'--*',label=r'$m = $%d' %M[k])
            k+=1

    # Plot solutions
    fig.legend()
    plt.xlabel(r'$t$')
    plt.ylabel(r'$x(t)$')
    plt.title(r'$\delta =$ %.0e' %delta[i])
    plt.grid()
    plt.show()

# Plotting convergence
fig = plt.gca()
for i in range(delta.size):
    fig.plot(convergence[:,i],'--*',label=r'$\delta = $%.0e' %delta[i])
fig.legend()
plt.xlabel('Iterations')
plt.ylabel(r'$|x-x^{\delta,\alpha}|_{2}$')
plt.title('Convergence')
plt.grid()
plt.show()

# Print table
print('--------------------------------------------------------')
title = 'm     '
for i in range(delta.size):
    title = title + 'd = %.0e  ' %delta[i]
print(title)
print('--------------------------------------------------------')
for j in range(M.size):
    row = '%3d   ' %M[j]
    for i in range(delta.size):
        row = row + '%.1e    ' %table[i,j]
    print(row)
print('--------------------------------------------------------')