# Loading libraries
import numpy as np

# Independent variable and bound
t = np.linspace(0,1,50)

# Variables of Kx = y
x = -1/2*np.exp(-2*t)
y = np.exp(-2*t)

# Noise
delta = 1e-6
noise = delta*np.sin(t/delta**2)
yh = y + noise

# Derivating yh
xh = np.diff(yh)/(t[1]-t[0])

# Computing error - supremum norm
ey = np.amax(np.abs(y-yh))
ex = np.amax(np.abs(x[1:]-xh))

# Printing results
print 'Error of the data: ' + '%e' % ey
print 'Error of the solution: ' + '%e' % ex