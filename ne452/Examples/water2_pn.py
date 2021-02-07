from __future__ import print_function

from simtk.openmm import app
import simtk.openmm as mm
from simtk import unit
import sys
from sys import argv
import mdtraj
import mdtraj.reporters
import numpy as np

#####Parameters

steps = int(argv[1])
skipSteps = int(argv[2])

#####

pdb = app.PDBFile("water2.pdb")

forcefield = app.ForceField('amber10.xml', 'tip3p.xml')

nonbonded = app.CutoffNonPeriodic

#Use the next line for flexible waters
system = forcefield.createSystem(pdb.topology, nonbondedMethod=nonbonded, nonBondedCutoff=1e3*unit.nanometer, rigidWater=False)
#Use the next line for rigid/constrained waters
#system = forcefield.createSystem(pdb.topology, nonbondedMethod=nonbonded, nonBondedCutoff=1e3*unit.nanometer, rigidWater=True)

dt = 0.5 * unit.femtoseconds

integrator = mm.LangevinIntegrator(30*unit.kelvin, 1.0/unit.picoseconds,dt)

#Use the next line for the Reference platform, slow, easier to read, will only use 1 core
platform = mm.Platform.getPlatformByName('Reference')
#Use the CPU platform, faster, can use multiple cores primarily to do non-bonded interactions (fft's in parallel, etc)
#platform = mm.Platform.getPlatformByName('CPU')
simulation = app.Simulation(pdb.topology, system, integrator, platform)
simulation.context.setPositions(pdb.positions)
simulation.context.computeVirtualSites()
state = simulation.context.getState(getForces=True, getEnergy=True, getPositions=True)
potential_energy = state.getPotentialEnergy()
potential_energy.in_units_of(unit.kilocalorie_per_mole)

kilocalorie_per_mole_per_angstrom = unit.kilocalorie_per_mole/unit.angstrom
for f in state.getForces():	
	print(f.in_units_of(kilocalorie_per_mole_per_angstrom))

from simtk.openmm import LocalEnergyMinimizer
LocalEnergyMinimizer.minimize(simulation.context, 1e-1)
	
simulation.context.setVelocitiesToTemperature(30*unit.kelvin)
#simulation.step(10)

#Outputs progress to command line
simulation.reporters.append(app.StateDataReporter(sys.stdout, skipSteps, step=True, 
    potentialEnergy=True, temperature=True, progress=True, remainingTime=True, 
    speed=True, totalSteps=steps, separator='\t'))

#Saves trajectory to .pdb file that can be opened in VMD
simulation.reporters.append(app.PDBReporter('trajectory.pdb', skipSteps))
#Saves trajectory file to binary format
simulation.reporters.append(mdtraj.reporters.HDF5Reporter('water2.h5', skipSteps))

#Performs the simulation
simulation.step(steps)

#Close binary trajectory
simulation.reporters[2].close()

#Read the output file
output_file = mdtraj.formats.HDF5TrajectoryFile('water2.h5')
data = output_file.read()
positions = data.coordinates
potE = data.potentialEnergy

#Check the distance between 1st H and O atoms 
for i in range(len(positions)):
    print(np.linalg.norm(positions[i][0] - positions[i][1]))




