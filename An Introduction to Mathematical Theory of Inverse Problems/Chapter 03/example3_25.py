# Loading libraries
import numpy as np
import matplotlib.pyplot as plt

# Main parameters
n = 32
delta = 1e-3
def xfun(t):
    return t**2
def yfun(t):
    return 1/3*t**3

# Auxiliar functions
def kfun(t,s):
    fk = np.zeros(s.size)
    fk[s<=t] = 1.
    return fk
def computeA(n):
    A = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            A[i,j] = 1/n*np.min(np.array([i+1,j+1]))
    return A
def computeinverse(n):
    A = n*(np.diag(np.repeat(2,n))+np.diag(np.repeat(-1,n-1),k=-1)
        +np.diag(np.repeat(1,n-1),k=1))
    A[-1,-1] = 1
    return A
                   
t = np.arange(1,n+1)/n # Independent variable array
h = 1/n # Discretization of the independent variable
x = xfun(t) # Unknown dependent variable
y = yfun(t) # Known dependent variable

# Add noise
if delta == 0:
    yd = np.copy(y)
else:
    yd = np.zeros(y.size)
    for i in range(y.size):
        yd[i] = np.random.normal(loc=y[i],scale=.21*delta)

# Linear system of Collocation Method
beta = np.copy(yd)
A = computeA(n)
alpha = np.linalg.solve(A,beta)

# Compute xn
xn = np.zeros(x.size)
for i in range(x.size):
    for j in range(x.size):
        xn[i] = xn[i] + alpha[j]*kfun(t[j],t[i])

# Print error
error = np.sqrt(1/n*np.sum((x-xn)**2))
print('COLOCATION METHOD')
print('n = %d' %n)
print('delta = %.1e' %delta)
print('Error: %.2e' %error)

# Plot solution
plt.plot(t,x)
plt.plot(t,xn,'--*')
plt.grid()
plt.xlabel(r'$t$')
plt.ylabel(r'$x(t)$')
plt.title('Collocation Method')
plt.legend(('Exact',r'$x^{\alpha,\delta}$'))
plt.show()