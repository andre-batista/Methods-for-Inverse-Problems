# Loading libraries
import numpy as np
import matplotlib.pyplot as plt

# Samples of time
ts = np.array([.0,.25,.5,.75,1.])

# Discretization models
N = np.array([4,8,16,32])

# Error variable
error = np.zeros((ts.size,N.size))

# Compute error for each discretization
ni = 0
for n in N:

    # Time array
    t = np.linspace(0,1,n+1)

    # Size of cell
    h = 1/n

    # Correct solution
    x = np.exp(t)

    # Data input
    y = (np.exp(t+1)-1)/(t+1)
    
    # Build coefficient matrix
    A = np.zeros((n+1,n+1))
    for i in range(n+1):
        A[i,0] = .5
        A[i,-1] = .5*np.exp(i*h)
        for j in range(1,n):
            A[i,j] = np.exp(i*j*h*h)
    A = h*A

    # Solve linear system
    xh = np.linalg.solve(A,y)

    # Computing error
    j = 0
    for i in range(t.size):
        if t[i] == ts[j]:
            error[j,ni] = x[i]-xh[i]
            j+= 1
    ni+=1

    plt.plot(t,xh,'-*')

# Print results
title = 't       '
for n in N:
    title = title + 'n = %d   ' %n
print(title)
for i in range(ts.size):
    row = '%.2f    ' % ts[i]
    for j in range(N.size):
        row = row + '%.2f    ' %error[i,j]
    print(row)

# Plot results
plt.plot(t,x)
plt.legend(('n = 4','n = 8','n = 16','n = 32','x(t)'))
plt.xlabel('t')
plt.ylabel('x(t)')
plt.title('Example 1.15')
plt.show()