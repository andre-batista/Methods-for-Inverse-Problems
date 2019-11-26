# Load libraries
import numpy as np
import matplotlib.pyplot as plt

# Define L2 norm function
def norml2 (y,dt):
    return np.sqrt(np.sum(np.abs(y)**2)*dt)

# Define regularization method
def regularization (y,t,h):
    Rh = np.zeros(t.shape)
    for i in range(0,t.size):
        if 0 <= t[i] and t[i] < h/2:
            Rh[i] = 1/h*(4*np.interp(t[i]+h/2,t,y)-np.interp(t[i]+h,t,y)-3*y[i])
        elif h/2 <= t[i] and t[i] <= 1-h/2:
            Rh[i] = 1/h*(np.interp(t[i]+h/2,t,y)-np.interp(t[i]-h/2,t,y))
        else:
            Rh[i] = 1/h*(3*y[i]+np.interp(t[i]-h,t,y)-4*np.interp(t[i]-h/2,t,y))
    return Rh

# Discretization size
n = 50

# Time array
t = np.linspace(0,1,n+1)
dt = t[1]-t[0]

# x (testbench) and y variables
x = np.exp(2*t-1)
y = .5*np.exp(2*t-1)
E = norml2(np.diff(np.diff(x)/dt)/dt,dt)

# Add noise
yd = y + .05*y*np.random.rand(y.size)
delta = np.linalg.norm(y-yd)

# Regularization parameter
h = np.linspace(1e-4,.5,100)

# Compute residual of approximation
fh = np.zeros(h.shape)
for i in range(h.size):
    fh[i] = np.linalg.norm(x-regularization(yd,t,h[i]))

# Optimal parameter
h_star = (delta/E)**(1/3)
x_star = regularization(yd,t,h_star)
fh_star = np.linalg.norm(x-x_star)

# The Cauchy-Schwarz inequality
lhs = norml2(regularization(y,t,h_star),dt)
rhs = norml2(x,dt)
c = rhs/lhs
print('CHAUCHY-SCHWARZ INEQUALITY')
print('Left-hand side is: %.2e' %lhs)
print('Right-hand side is %.2e' %rhs)
print('c value for equality condition is: %.2e' %c)

# Plot recovered function
plt.plot(t,x)
plt.plot(t,x_star,'--*')
plt.legend((r'$x$',r'$x^{\delta,h}$'))
plt.grid()
plt.xlabel(r'$t$')
plt.ylabel(r'$x(t)$')
plt.title('Function recovering')
plt.show()

# Plot residual minimization
plt.plot(h,fh,'--*')
plt.plot(h_star,fh_star,'or')
plt.xlabel(r'$h$')
plt.ylabel(r'$||R_{h(\delta)}y^{\delta}-x||$')
plt.title('Optimal parameter')
plt.grid()
plt.show()