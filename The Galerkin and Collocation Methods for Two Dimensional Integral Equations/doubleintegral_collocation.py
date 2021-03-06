# Library for matrix computation
import numpy as np

# Library for plotting surface figures 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

m = 10 # Number of collocation points in x-axis
n = 10 # Number of collocation points in y-axis

# Input function
def gfun(x,y):
    return (np.exp(x)-1)*(np.exp(y)-1)
    # return x**2*y/2 + y**2*x/2
# Exact solution
def ffun(x,y):
    return np.exp(x+y)
    # return x+y
# Base function
def kfun(x,y,u,v):
    if x.size > u.size:
        k = np.zeros(x.shape)
    else:
        k = np.zeros(u.size)
    k[np.logical_and(u<=x,v<=y)] = 1
    return k

# Coordinates and grid
x = np.arange(1,m+1)/m
y = np.arange(1,n+1)/n
xv, yv = np.meshgrid(x, y)

# Evaluating the input and exact solution on the grid
f = ffun(xv,yv)
g = gfun(xv,yv)

# Compute coefficient matrix for A*alpha=beta
A = np.zeros((m*n,m*n))
for u in range(m*n):
    i,j = np.unravel_index([u],(m,n))
    for v in range(m*n):
        p,q = np.unravel_index([v],(m,n))
        A[u,v] = min([i+1,p+1])*min([j+1,q+1])/m/n

# Compute beta array 
beta = np.copy(g.reshape(m*n))

# Solve system for alpha
alpha = np.linalg.solve(A,beta)

# Evaluate the solution on the grid
fa = np.zeros((m,n))
for i in range(m):
    for j in range(n):
        fa[i,j] = np.sum(alpha*kfun(xv.reshape(m*n),yv.reshape(m*n),x[i],y[j]))

# Compute error
error = 1/m/n*np.sum(np.abs((f-fa)/f))
print('Error: %.2e' %error)

# Plot exact and recovered solution
fig = plt.figure()
ax = fig.add_subplot(1, 2, 1, projection='3d')
surf1 = ax.plot_surface(xv, yv, f, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
fig.colorbar(surf1, shrink=0.5, aspect=5)
ax.set_xlabel(r'$x$')
ax.set_ylabel(r'$y$')
ax.set_zlabel(r'$f(x,y)$')
plt.title('Exact Solution')
plt.grid()
ax = fig.add_subplot(1, 2, 2, projection='3d')
surf2 = ax.plot_surface(xv, yv, fa, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
fig.colorbar(surf2, shrink=0.5, aspect=5)
ax.set_xlabel(r'$x$')
ax.set_ylabel(r'$y$')
ax.set_zlabel(r'$f(x,y)$')
plt.title('Recovered Solution')
plt.grid()
plt.show()