# Classical trajectory for a 1d harmonic oscillator
import numpy as np
import matplotlib.pyplot as plt
from sys import argv
#initial conditions
q=1. # position
v=0. # velocity
mass=1.
k=1 # force constant
algo=argv[1] # integration algorithm: 'Euler' or 'VelocityVerlet'
dt=float(argv[2]) # time step
nsteps=int(argv[3]) # number of time steps
# Euler's method
traj=np.zeros((nsteps,4),float) # array for storing trajectory data
for i in range(nsteps):
	E=.5*mass*v*v+.5*k*q*q # total energy
	if (algo=='Euler'):
		q=q+dt*v
		F= - k*q
		a=F/mass
		v=v+dt*a
	if (algo=='VelocityVerlet'):
		F= - k*q
		a=F/mass
		v=v+.5*a*dt
		q=q+v*dt
		F= - k*q
		a=F/mass
		v=v+.5*a*dt
# save trajection data in traj array
	traj[i,0]=i*dt # current time
	traj[i,1]=q   # position
	traj[i,2]=mass*v   # momentum
	traj[i,3]=E # total energy
# some data analysis
qvalues=traj[:,1]
plt.hist(qvalues, bins =50,density=True)
plt.xlabel('q')
plt.ylabel('Probability')
plt.title('Histogram of q')
plt.savefig("q_histo"+algo+str(dt)+".png")

#clear figure and axes
plt.clf()
plt.cla()

Evalues=traj[:,3]
plt.hist(Evalues, bins =50,density=True)
plt.xlabel('E')
plt.ylabel('Probability')
plt.title('Histogram of E')
plt.savefig("E_histo"+algo+str(dt)+".png")

# phase space plot
plt.clf()
plt.cla()

plt.plot(traj[:,1],traj[:,2])
plt.xlabel('q')
plt.ylabel('p')
plt.title('phase space plot')
plt.savefig("pq"+algo+str(dt)+".png")

# trajectory
plt.clf()
plt.cla()

plt.plot(traj[:,0],traj[:,1])
plt.xlabel('t')
plt.ylabel('q')
plt.title('q trajectory')
plt.savefig("qt"+algo+str(dt)+".png")

plt.clf()
plt.cla()

plt.plot(traj[:,0],traj[:,3])
plt.xlabel('t')
plt.ylabel('E')
plt.title('E trajectory')
plt.savefig("Et"+algo+str(dt)+".png")

