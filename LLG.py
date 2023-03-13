import time
import numpy as np
from numpy.random import default_rng
rng = default_rng()
from datetime import datetime
import os
import matplotlib.pyplot as plt
from LLG_support_functions import * 




#
# Model parameters
#
J0 = 2.25 # exchange interaction in the ab-plane (meV)
K0 = 0.09 # exchange anisotropy (z-axis) in the ab-plane (meV)
K1 = 0.0 # single-ion anisotropic energy (meV) 
theta = 1.05 # tilting angle 


#
# initialization of Hamiltonian and spins
#
H0 = Hamiltonian(theta, J0, K0, K1)
S0 = spin("random", H0.N0)



##############################################################
#
# example 1: energy minimization
#
##############################################################
now = datetime.now()
dt_string = now.strftime("%Y-%d-%m--%H-%M-%S")
os.mkdir(dt_string)


Nt_min =300
Bz_sweeps = np.linspace(-0.4, 0.4, num=5)
sigma = 20

for ii, Bz in enumerate(Bz_sweeps):
	H0.update_B_field([0,0,Bz])
	S0 = S0.Gaussian_move(sigma) 
	S0.M = H0.SIM_minimization_steps(S0, Nt_min, verbose = False)
	S0.plot_spins(filename=dt_string+'/' + f'{ii}.pdf')
	print(S0.get_R())

exit()

##############################################################
#
# example 2: Monte Carlo
#
##############################################################
now = datetime.now()
dt_string = now.strftime("%Y-%d-%m--%H-%M-%S")
os.mkdir(dt_string)


kbT = 0.01
sigma = 60
H0.update_B_field([0,0,-0.2])
for step in range(100):
	# S0.M, accept_ratio = H0.single_MC_step(S0, kbT, "Gaussian")
	S0.M, accept_ratio = H0.single_MC_step(S0, kbT, "adapted_Gaussian", sigma=sigma)
	print(H0.get_energy(S0.M), accept_ratio, sigma)
	S0.plot_spins(filename=dt_string+'/' + f'{step}.pdf')
	sigma = min(60, 0.5/(1-accept_ratio)*sigma)


##############################################################
#
# example 3: Real time evolution by LLG
#
##############################################################
now = datetime.now()
dt_string = now.strftime("%Y-%d-%m--%H-%M-%S")
os.mkdir(dt_string)


integrator = "RK45" # integrator. "Heun",  "RK45", "Depondt", "VP", "Ivanov"
dt = 0.01
gamma = 2.0 # gyromagnetic ratio
alpha = 1.0 # Gilbert damping constant

Z1 = gamma/(1+alpha**2)
Z2 = Z1*alpha/muS
Zs = np.array([Z1, Z2])

# parameter for sweeping
t_fin = 40 # final time
Nt_fin = int(t_fin/dt)
time_steps = np.arange(Nt_fin) 

#
# Prepare sweeping protocol for B-field 
#
B_sweep = np.zeros((Nt_fin, 3))
B_sweep[:,2] = np.linspace(0, 1, num=Nt_fin)


#
# time evolution for time-varying B-field
#
obs_file = dt_string + '/' + "data.txt"
f = open(obs_file, "a")
f.write("%time	Bz	Mz \n")

for ii in range(Nt_fin):

	H0.update_B_field(B_sweep[ii,:])
	
	if integrator == "Heun": 
		S0.M = H0.Heun_step(S0.M, dt, Zs)
	elif integrator == "RK45":
		S0.M = H0.RK45_step(S0.M, dt, Zs)

	if ii%1000 == 0:
		current_time = ii*dt
		print(f'current time ={current_time}')
		fig_name = "./" + dt_string + "/" + f"t={current_time}.pdf"
		S0.plot_spins(filename = fig_name)

	if ii%100 == 0:
		current_time = ii*dt
		Bz = B_sweep[ii,2]
		Mz = S0.M[...,2].mean()
		f.write(f"{current_time}, {Bz}, {Mz} \n")

f.close()