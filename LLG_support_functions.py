import numpy as np
from numpy.random import default_rng
rng = default_rng()
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import math
import matplotlib.transforms as mtransforms

muS = 1.5 # magnetization on Cr atoms (muB)
a0 = 6.83 # lattice consant of a hexagonal lattice (angstrom) 
d0 = 6.83/np.sqrt(3) # distance between neighboring sites of a hexagonal lattice (angstrom) 

a1 = a0*np.array([-1/2, np.sqrt(3)/2, 0]) # Bravais lattice vector 
a2 = a0*np.array([1/2, np.sqrt(3)/2, 0]) # Bravais lattice vector
a3 = np.array([0,0, 6.61]) # Bravais lattice vector


def get_atom_position(l1, l2, l3, b):
	return l1*a1+l2*a2+l3*a3+b*[0,d0,0]


#
# Hamiltonian class 
#
class Hamiltonian:
	def __init__(self, theta, J0, K0, K1, *, relaxed = 1.0, J_perp_scale=1.0):
		self.theta = theta
		theta_rad = 2*math.pi/360*theta # tilted angle in radian
		if theta == 0:
			self.N0 = 50
		else:
			self.N0 = int(1/theta_rad) # approximate Moire periodicity
		self.J0 = J0
		self.K0 = K0
		self.K1 = K1
		self.Bext = np.zeros((3,))
		self.J_perp_scale = J_perp_scale
		self.relaxed = relaxed

		J_perp_rigid = set_inter_layer_coupling('deltaE_Nano_2.csv', self.N0)
		J_perp_relax = set_inter_layer_coupling('deltaE_Nano_3.csv', self.N0)
		self.J_perp = (J_perp_rigid*(1-relaxed) + J_perp_relax*relaxed)*self.J_perp_scale
		if theta == 0:
			self.J_perp = self.J_perp[0,0,0,0]
		self.total_sites = 4*self.N0*self.N0*2

	def update_B_field(self, B_new):
		self.Bext = B_new

	#
	# get the effective magnetic fields
	#
	def get_Heff(self, M0):

		N3, N1, N2, _, _ = np.shape(M0)
		spin_a = M0[...,0,:] # spins of sublattice A
		spin_b = M0[...,1,:] # spins of sublattice B

		Heff = np.zeros((N3,N1,N2,2,3))
		Heff[...,0,:] = self.J0*(spin_b + np.roll(spin_b, 1, axis=1) + np.roll(spin_b, 1, axis=2)) # confusing about the roll direction, roll = shift
		Heff[...,1,:] = self.J0*(spin_a + np.roll(spin_a, -1, axis=1) + np.roll(spin_a, -1, axis=2))
		
		Heff[0,...] += np.multiply(self.J_perp[0,:,:,:,None], M0[1,...]) # bottom layer
		Heff[-1,...] += np.multiply(self.J_perp[-1,:,:,:,None], M0[-2,...]) # top payer

		for ii in range(1, N3-1): # = range(1,3) = (1,2)
			Heff[ii,...] += np.multiply(self.J_perp[ii-1,:,:,:,None], M0[ii-1,...]) + np.multiply(self.J_perp[ii,:,:,:,None], M0[ii+1,...])

		# anisotropy term
		Heff[...,2] += 2*self.K1*M0[...,2]

		Heff[...,0,2] += self.K0*(spin_b[...,2] + np.roll(spin_b[...,2], 1, axis=1) + np.roll(spin_b[...,2], 1, axis=2)) 
		Heff[...,1,2] += self.K0*(spin_a[...,2] + np.roll(spin_a[...,2], -1, axis=1) + np.roll(spin_a[...,2], -1, axis=2))


		# external field
		Heff[...,0] += self.Bext[0]
		Heff[...,1] += self.Bext[1]
		Heff[...,2] += self.Bext[2]

		return Heff


	#
	# the right-hand side of the LLG equation, M-dot
	#
	def dM(self, M0, Zs):
		Z1, Z2 = Zs
		tau = self.get_torque(M0)
		g0 = self.get_MMH(M0)
		dM = -Z1*tau - Z2*g0
		return dM

	#
	# time-evolution by the Heun scheme 
	#
	def Heun_step(self, M0, dt, Zs):
		# predictor
		Mp = M0 + dt*self.dM(M0, Zs)
		# corrector
		Mc = M0 + 0.5*dt*(self.dM(M0, Zs) + self.dM(Mp, Zs))

		return Mc

	#
	# time-evolution by the RK45
	#
	def RK45_step(self, M0, dt, Zs):
		k1 = self.dM(M0, Zs)
		k2 = self.dM(M0 + 0.5*dt*k1, Zs)
		k3 = self.dM(M0 + 0.5*dt*k2, Zs)
		k4 = self.dM(M0 + dt*k3, Zs)
		
		M1 = M0 + 1.0/6.0*dt*(k1+2*k2+2*k3+k4)

		return M1



	#
	# cross product of M and H
	#
	def get_torque(self, M0):
		H0 = self.get_Heff(M0)
		tau = np.empty_like(M0)

		tau[...,0] = M0[...,1]*H0[...,2]- M0[...,2]*H0[...,1]
		tau[...,1] = M0[...,2]*H0[...,0]- M0[...,0]*H0[...,2]
		tau[...,2] = M0[...,0]*H0[...,1]- M0[...,1]*H0[...,0]

		return tau


	#
	# get the steepest descent with conserved norm: Mx(MxH) = (M.H)M-M^2H, Eq.(3.25), (3.26) in Exl's thesis
	#
	def get_MMH(self, M0):
		H0 = self.get_Heff(M0)
		g0 = np.empty_like(H0)
				
		# muS_list = np.linalg.norm(M0[:,0,0,0,:], axis =1)
		# for ii in range(4):
		# 	g0[ii,...] = -H0[ii,...]*muS_list[ii]**2

		muS_squared = np.linalg.norm(M0, axis = -1, keepdims=True)**2
		g0 = -muS_squared*H0 # array broadcasting

		MH = (M0[...,2]*H0[...,2] + M0[...,1]*H0[...,1]+ M0[...,0]*H0[...,0]) 
		g0[...,0] +=  M0[...,0]*MH
		g0[...,1] +=  M0[...,1]*MH
		g0[...,2] +=  M0[...,2]*MH

		return g0



	#
	# get energy
	#
	def get_energy(self, M0):

		H0 = self.get_Heff(M0)
		energy = 0.0
		N3, N1, N2, _, _ = np.shape(M0)

		energy = -np.sum(H0*M0)/2

		Zeemann = np.sum(M0*self.Bext) # broadcasting the arrays
		
		energy = energy - Zeemann/2
		energy = energy/self.total_sites

		return energy




	#
	# semi-implicite method with fixed B
	#
	def SIM_minimization_steps(self, S0, Nt_min, verbose=False):
		tol = 10.0**(-8)

		# Note that changing M0 now changes S0.M
		M0 = S0.M
		
		dt = 1/np.sum(M0*self.get_Heff(M0))
		M1 =self.SIM_update_step(M0, dt)
		
		tt = 0
		g0 = 1.0
		last_energy = 10.0*np.ones(20)
		e1 = self.get_energy(M1)

		while g0 > tol or tt < Nt_min:

			dt = self.get_backtracking_time_steps(M0, 0.000001, 0.7)

			M0 = M1
			M1 = self.SIM_update_step(M0,dt)
			e0 = e1
			e1 = self.get_energy(M1)

			last_energy[tt%20] = e0

			g0 = np.amax(abs(self.get_MMH(M1)))

			if tt%50 == 0:
				norm_error = S0.get_norm_error()
				if  norm_error> 0.01:
					print('norm is not conserved')
					exit()
				if verbose == True:
					print('g0, energy, dts = ', g0, self.get_energy(M0), dt)

			if abs(e1-e0) < 10.0**(-8) and tt > 10 and g0 < 0.01:
				print('stopped at tt= ', tt)
				print('e1-e0 = ', e1 - e0)
				print('g0 = ', g0)
				print('')
				break

			tt = tt + 1
		
		print('finished SIM minimization')
		return M0







	#
	# Semi-implicit method to get M_k+1 from M_k with a time step dt: (see Eq. 3.34 of Exl's thesis)
	#
	def SIM_update_step(self, M0, dt):

		MH = self.get_torque(M0)

		Nr = 4+dt**2*(MH[...,0]**2+MH[...,1]**2+MH[...,2]**2)

		M1 = np.empty_like(M0)

		M1[...,0] = 4*M0[...,0] \
			+4*dt*(MH[...,1]*M0[...,2]-MH[...,2]*M0[...,1]) \
			-dt**2*M0[...,0]*(-MH[...,0]**2+MH[...,1]**2+MH[...,2]**2) \
			+2*dt**2*MH[...,0]*(MH[...,1]*M0[...,1]+MH[...,2]*M0[...,2])

		M1[...,1] = 4*M0[...,1] \
			+4*dt*(MH[...,2]*M0[...,0]-MH[...,0]*M0[...,2]) \
			-dt**2*M0[...,1]*(MH[...,0]**2-MH[...,1]**2+MH[...,2]**2) \
			+2*dt**2*MH[...,1]*(MH[...,0]*M0[...,0]+MH[...,2]*M0[...,2])

		M1[...,2] = 4*M0[...,2] \
			+4*dt*(MH[...,0]*M0[...,1]-MH[...,1]*M0[...,0]) \
			-dt**2*M0[...,2]*(MH[...,0]**2+MH[...,1]**2-MH[...,2]**2) \
			+2*dt**2*MH[...,2]*(MH[...,1]*M0[...,1]+MH[...,0]*M0[...,0])

		M1[...,0] = M1[...,0]/Nr
		M1[...,1] = M1[...,1]/Nr
		M1[...,2] = M1[...,2]/Nr

		return M1


	#
	# Armijo condition + backtracking to find the time step size
	#
	def get_backtracking_time_steps(self, M0, xi, eta):

		dt = 1.0
		tt = 1
		e0 = self.get_energy(M0)
		g0 = self.get_MMH(M0)
		df2 = np.sum(g0**2)

		M1 = self.SIM_update_step(M0, dt)
		e1 = self.get_energy(M1)

		while e1 + xi*dt*df2 >= e0:
			dt = dt* eta
			M1 = self.SIM_update_step(M0, dt)
			e1 = self.get_energy(M1)
			tt = tt +1
			# print(tt, dt, e0, e1, xi*dt*df2)

		return dt



	#
	# Single Monte Carlo step with temperature kbT 
	#
	def single_MC_step(self, S0, kbT, update_method, *, sigma = None):
		M0 = S0.M.copy()

		if update_method == 'random':
			S1 = spin('random', self.N0)
		elif update_method == 'Gaussian':
			sigma = 2/25 * (kbT)**0.2
			S1 = S0.Gaussian_move(sigma)
		elif update_method == 'adapted_Gaussian' and sigma:
			S1 = S0.Gaussian_move(sigma)
		else:
			print('check input arguments for single_MC_step')
			exit()
	
		M1 = S1.M.copy()

		prob = rng.random(self.total_sites)
		accept_count = 0
		for lin_index in range(self.total_sites):
			position = np.unravel_index(lin_index, (4, self.N0, self.N0, 2))
			x3, x1, x2, s = position
			spin_new = M1[x3, x1, x2, s, :]
			delE = self.get_single_site_energy_difference(M0, position, spin_new)
			
			if delE < 0:
				M0[x3,x1,x2,s,:] = spin_new
				accept_count += 1
			elif prob[lin_index] < np.exp(-delE/kbT):
				M0[x3,x1,x2,s,:] = spin_new
				accept_count += 1
		
		accept_ratio = accept_count/self.total_sites

		return M0, accept_ratio

	#
	# get the energy difference between the new spin and current spin on the i-th site. note that the energy is linear in M_i except the singe-ion anisotropy 
	#
	def get_single_site_energy_difference(self, M0, position, spin_new):

		x3, x1, x2, s = position 
		_, N1, N2, _, _ = np.shape(M0)

		spin_old = M0[x3,x1,x2,s,:]
		assert abs(np.linalg.norm(spin_new) - np.linalg.norm(spin_old)) < 0.001
		delM = spin_new - spin_old
		delE = 0.0

		# in-plane NN
		if s ==0:
			M_nn = M0[x3,x1,x2,1,:]+ M0[x3,x1-1,x2,1,:] + M0[x3, x1,x2-1,1,:]
		else:
			M_nn = M0[x3,x1,x2,0,:]+ M0[x3,int(x1+1)%N1,x2,0,:] + M0[x3, x1,int(x2+1)%N2,0,:]

		delE += -self.J0*np.dot(delM, M_nn)
		delE += -self.K0*delM[2]*M_nn[2]


		# interlayer 
		if x3 == 0:
			delE += -self.J_perp[0,x1,x2,s]*np.dot(delM, M0[1,x1,x2,s,:])
		elif x3 == 3:
			delE += -self.J_perp[2,x1,x2,s]*np.dot(delM, M0[2,x1,x2,s,:])
		else:
			delE += -self.J_perp[x3-1,x1,x2,s]*np.dot(delM, M0[x3-1,x1,x2,s,:]) - self.J_perp[x3,x1,x2,s]*np.dot(delM, M0[x3+1,x1,x2,s,:])

		# single-ion anisotropy term
		delE += -self.K1*(spin_new[2]**2-spin_old[2]**2)

		# external field
		delE += -np.dot(delM, self.Bext)

		return delE

		
#
# set up interlayer coupling 
#
def set_inter_layer_coupling(filename, N0):
	#
	# read delE extracted from Nano. letter
	#
	def read_delE(filename):
		d = np.genfromtxt(filename, delimiter=',')
		r = np.linspace(0,1,19)
		delE_diag = d[:,0]
		delE_edge = d[:,1]

		# initial and final points of lines of data points
		x0 = np.array([0,0,1,0,1,0,0,1])
		y0 = np.array([0,1,0,1,1,0,1,0])
		x1 = np.array([1,1,0,0,1,1,1,-1])
		y1 = np.array([0,1,1,0,0,1,-1,1])
		data_type = ['e','e', 'e', 'e', 'e', 'd', 'd', 'd']

		x = np.array([])
		y = np.array([])
		z = np.array([])

		for ii in np.arange(8):
			x_vec = np.linspace(x0[ii], x1[ii], 19)
			y_vec = np.linspace(y0[ii], y1[ii], 19)

			if ii == 6 or ii == 7:
				x_vec = x_vec - np.floor(x_vec)
				y_vec = y_vec - np.floor(y_vec)
			
			if data_type[ii] == 'e':
				z_vec = delE_edge
			elif data_type[ii] == 'd':
				z_vec = delE_diag
			
			x = np.append(x, x_vec)
			y = np.append(y, y_vec)
			z = np.append(z, z_vec)

		points = np.vstack((x,y)).T 

		return points, z
	#
	# fitting of the imported delE by harmonics 
	#
	def J_fit(r, *args):
		#reciprocal vectors
		G1 = 2*math.pi*np.array([1,-1/math.sqrt(3)])
		G2 = 2*math.pi*np.array([0,2/math.sqrt(3)])
		G3 = -G1 - G2

		a1 = np.array([1,0])
		a2 = np.array([1/2, math.sqrt(3)/2])

		A = np.array([[1,1/2],[0,math.sqrt(3)/2]])

		# center of rotation
		r0 = np.expand_dims(a1/3+a2/3, axis = 1)

		r1 = np.dot(A, r)-r0

		# there are (4m+1) variables
		m = int((len(args)-1)/4)
		cp = args[:m]
		tp = args[m:2*m]
		cm = args[2*m:3*m]
		tm = args[3*m:4*m]
		amplitude = args[-1]

		Gs = [[G1, G2, G3],[G1-G2, G3-G1, G2-G3],[2*G1, 2*G2, 2*G3], [2*G1-G2, 2*G3-G1, 2*G2-G3], [G1-2*G2, G3-2*G1, G2-2*G3], [2*G1-2*G2, 2*G3-2*G1, 2*G2-2*G3], [3*G1, 3*G2, 3*G3],[3*G1-G2, 3*G3-G1, 3*G2-G3],[G1-3*G2, G3-3*G1, G2-3*G3],[3*G1-3*G2, 3*G3-3*G1, 3*G2-3*G3]]

		for ii in range(m):
			amplitude += cp[ii]*np.cos(np.dot(Gs[ii][0], r1) + tp[ii]) \
						+cp[ii]*np.cos(np.dot(Gs[ii][1], r1) + tp[ii]) \
						+cp[ii]*np.cos(np.dot(Gs[ii][2], r1) + tp[ii]) \
						+cm[ii]*np.cos(-np.dot(Gs[ii][0], r1) + tm[ii]) \
						+cm[ii]*np.cos(-np.dot(Gs[ii][1], r1) + tm[ii]) \
						+cm[ii]*np.cos(-np.dot(Gs[ii][2], r1) + tm[ii]) \

		return amplitude
	#
	# read the file
	#
	ratio_delE_J = -4*(1.5**2)
	points, z = read_delE(filename)

	J_perp = np.zeros((3,N0,N0,2))
	J_perp[0,...] =  z[6]/ratio_delE_J # J_perp of AB' (monoclinic) stacking: AFM (J < 0)
	J_perp[-1,...] =  z[6]/ratio_delE_J # J_perp of AB' (monoclinic) stacking: AFM (J < 0)
	
	#
	# fitting
	#
	harmonics = 3
	popt, pcov = curve_fit(J_fit, xdata=points.T, ydata=z, p0=[1]*(4*harmonics+5) )

	# 
	# grid for interpolation 
	# 
	N_grid = 3*N0+1
	r_grid = np.linspace(0,1, num = N_grid)
	grid_x, grid_y = np.meshgrid(r_grid, r_grid)
	r_grid = np.squeeze(np.array([[grid_x.reshape(N_grid**2,1)],[grid_y.reshape(N_grid**2,1)]]))

	fitData=J_fit(r_grid, *popt).reshape(N_grid, N_grid)
	# delE_interpolated = interpolate.griddata(points, z, (grid_x, grid_y), method='cubic').T # oddly the axis are swapped.

	J_perp[1,:,:,0] = fitData[:-1:3,:-1:3]/ratio_delE_J
	J_perp[1,:,:,1] = fitData[1:-1:3,1:-1:3]/ratio_delE_J
	return J_perp



#
# spin class
#
class spin:
	def __init__(self, initial_state, N0, *, dS = 0):
		self.M = np.zeros((4, N0, N0, 2, 3), dtype=float)
		self.N0 = N0
		self.dS = dS
		self.muS_list = np.array([muS + dS, muS, muS, muS + dS])

		if initial_state == "FM":
			self.M[...,0] = 0
			self.M[...,1] = 0
			self.M[...,2] = muS

		elif initial_state == "AFM":
			# odd layers are spin up
			self.M[0::2,:,:,:,0] = 0
			self.M[0::2,:,:,:,1] = 0
			self.M[0::2,:,:,:,2] = muS
			# even layers are spin down
			self.M[1::2,:,:,:,0] = 0
			self.M[1::2,:,:,:,1] = 0
			self.M[1::2,:,:,:,2] = -muS

		elif initial_state == "random":
			polar_angles = np.arccos(1-2*rng.random(4*N0*N0*2))
			azimuthal_angles = rng.random(4*N0*N0*2)*2*math.pi
			spin_spherical = np.zeros((4*N0*N0*2, 3))

			spin_spherical[:,0] = np.sin(polar_angles) * np.cos(azimuthal_angles)
			spin_spherical[:,1] = np.sin(polar_angles) * np.sin(azimuthal_angles)
			spin_spherical[:,2] = np.cos(polar_angles)
			self.M = muS*spin_spherical.reshape((4,N0,N0,2,3))

			assert abs(np.linalg.norm(self.M[0,0,0,0,:]) - muS) < 1.0E-10, f"spin normalization fails. {np.linalg.norm(self.M[0,0,0,0,:])}"

		self.renormalize_spins()



	#
	# renormalize the spins
	#
	def renormalize_spins(self):
		spin_shape = np.shape(self.M)
		spin_reshaped = np.reshape(self.M, (-1, 3))
		spin_norm = np.linalg.norm(spin_reshaped, axis = 1)

		spin_normalized = spin_reshaped/np.expand_dims(spin_norm, axis = 1)
		new_spins = np.reshape(spin_normalized, spin_shape)

		for ii in range(4):
			self.M[ii,...] = new_spins[ii,...]*self.muS_list[ii]


	#
	# create new spin with random Gaussian fluctuations Gamma with width sigma to spin M0
	#
	def Gaussian_move(self, sigma):
		S1 = spin("FM", self.N0, dS = self.dS)
		N3, N1, N2, s, _ = np.shape(self.M) 
		Gamma = rng.standard_normal((N3,N1,N2,s,3))
		S1.M = self.M + sigma*Gamma
		S1.renormalize_spins()
		return S1




	#
	# get the av. norm of the spins
	#
	def get_norm_error(self):
		ideal_norm = np.empty((4,self.N0,self.N0,2))

		for ii in range(4):
			ideal_norm[ii,...] = self.muS_list[ii]
		
		spin_norm = np.linalg.norm(self.M, axis = -1)

		error = np.amax(abs(ideal_norm-spin_norm))

		return error

	#
	# plotting spins. If filename is not given, plot will be shown.
	#
	def plot_spins(self, *, filename = None):
		spins = self.M[:,:,:,0,:]
		
		fig = plt.figure()
		spec = fig.add_gridspec(3, 4)
		Smax = max(self.muS_list)

		for spin_axis in range(3):
			for cols in range(4):
				ax = fig.add_subplot(spec[spin_axis,cols])
				transform = mtransforms.Affine2D().skew_deg(30, 0)

				im = ax.imshow(spins[cols,:,:,spin_axis].T, interpolation='none', origin='lower', \
						extent=[0, 1, 0, 1], clip_on=True, vmin = -Smax, vmax = Smax, cmap='bwr')

				trans_data = transform + ax.transData
				im.set_transform(trans_data)
				ax.axis('off')
				# display intended extent of the image
				x1, x2, y1, y2 = im.get_extent()
				ax.plot([x1, x2, x2, x1, x1], [y1, y1, y2, y2, y1],	transform=trans_data, alpha = 0.0)

				# im = ax.imshow(Ma[cols*N0:(cols+1)*N0,:].T, extent= (0,1,0,1), origin = 'lower', vmin = -1, vmax = 1, cmap='bwr')
				ax.tick_params(labelsize = 4.0)
				divider = make_axes_locatable(ax)
				cax = divider.append_axes('right', size='5%', pad=0.05)
				fig.colorbar(im, cax=cax, orientation='vertical')
				cax.tick_params(labelsize = 4.0)

		if  filename == None:
			plt.show()
		else:
			plt.savefig(filename)
			
		plt.close(fig)



	#
	# calculate the resistance
	#
	def get_R(self):
		T0_up = 0.245013
		T0_dn = 0.109081
		N0 = self.N0

		Min = self.M[:,:,:,0,:] # only the A-sites are taken, (4,N0,N0,3)

		M_3d_in = np.moveaxis(Min, -1, 0) # (3, 4, N0, N0)
		M_3d_norm = np.linalg.norm(M_3d_in, axis = 0)
		M_3d_n = M_3d_in/M_3d_norm
			
		cosbeta = np.zeros((3,N0,N0))

		cosbeta[0,:,:] = np.einsum('aij,aij->ij', M_3d_n[:,0,:,:], M_3d_n[:,1,:,:])
		cosbeta[1,:,:] = np.einsum('aij,aij->ij', M_3d_n[:,1,:,:], M_3d_n[:,2,:,:])
		cosbeta[2,:,:] = np.einsum('aij,aij->ij', M_3d_n[:,2,:,:], M_3d_n[:,3,:,:])

		T1_up = np.ones((N0,N0))*T0_up
		T1_dn = np.ones((N0,N0))*T0_dn

		T2_up = T0_up*0.5*((T1_up + T1_dn) + (T1_up - T1_dn)*cosbeta[0,:,:])
		T2_dn = T0_dn*0.5*((T1_up + T1_dn) - (T1_up - T1_dn)*cosbeta[0,:,:])

		T3_up = T0_up*0.5*((T2_up + T2_dn) + (T2_up - T2_dn)*cosbeta[1,:,:])
		T3_dn = T0_dn*0.5*((T2_up + T2_dn) - (T2_up - T2_dn)*cosbeta[1,:,:])

		T4_up = T0_up*0.5*((T3_up + T3_dn) + (T3_up - T3_dn)*cosbeta[2,:,:])
		T4_dn = T0_dn*0.5*((T3_up + T3_dn) - (T3_up - T3_dn)*cosbeta[2,:,:])

		G_total = np.sum(T4_up + T4_dn)

		R_total = 1/G_total

		return R_total*N0*N0	


	#
	# get wrapping numbers of a unit cell and three quadrants
	#
	def get_wrapping_numbers(self):
		
		M1 = np.copy(self.M[1,:,:,0,:])/self.muS_list[1] # spin on the 2nd layer
		M2 = np.copy(self.M[2,:,:,0,:])/self.muS_list[2] # spin on the 3rd layer
		N0 = self.N0
		N0_h = int(N0/2)
		delL = int(N0/12.0)
		
		M1 = np.roll(M1, (delL, delL), axis = (0,1))
		M2 = np.roll(M2, (delL, delL), axis = (0,1))

		M1_sub = [M1[:N0_h,:N0_h,:],M1[:N0_h,N0_h:,:],M1[N0_h:,:N0_h,:]]
		M2_sub = [M2[:N0_h,:N0_h,:],M2[:N0_h,N0_h:,:],M2[N0_h:,:N0_h,:]]

		W1 = get_skyrmion_number(M1)
		W2 = get_skyrmion_number(M2)
		
		W1_sub = np.zeros(3)
		W2_sub = np.zeros(3)

		for ii in range(3):
			W1_sub[ii] = get_skyrmion_number(M1_sub[ii], boundary = 1)
			W2_sub[ii] = get_skyrmion_number(M2_sub[ii], boundary = 1)

		return [W1, W1_sub, W2, W2_sub]





#
# calculate the skrymion number based on the analytical formula
#
def get_skyrmion_number(M1, *, boundary = 0):
	M1_new, delM1 = get_gradient_tri_lattice(M1, boundary = boundary)

	eijk = np.zeros((3, 3, 3))
	eijk[0, 1, 2] = eijk[1, 2, 0] = eijk[2, 0, 1] = 1
	eijk[0, 2, 1] = eijk[2, 1, 0] = eijk[1, 0, 2] = -1

	W1 = math.sqrt(3)/2*np.einsum('ijs, stu, ijt, iju -> ', M1_new, eijk, delM1[0,...], delM1[1,...])/(4*math.pi)
	
	return W1

#
# get gradient by nonconserving first order least square gradient estimation (see Mook's Ph.D thesis and )
# input array should be (N1, N2, s_dim) and along a1 and a2 direction defined in the code below
#
def get_gradient_tri_lattice(M1, *, boundary = 0):
	N1, N2, s_dim = np.shape(M1)

	a1 = np.array([1,0])
	a2 = np.array([1/2, math.sqrt(3)/2])
	
	D0 = np.array([a1, -a1, a2, -a2, -a1+a2, a1-a2])
	D1 = np.hstack((D0, [[1],[1],[1],[1],[1],[1]])) # (NN, a)

	g = np.zeros([3, N1,N2, s_dim]) # (a, i,j, s)

	delM1_NN = np.zeros([6,N1,N2, s_dim]) # (NN, i,j,s)
	delM1_NN[0,...] = np.roll(M1, -1, axis = 0) - M1
	delM1_NN[1,...] = np.roll(M1, 1, axis = 0) - M1
	delM1_NN[2,...] = np.roll(M1, -1, axis = 1) - M1
	delM1_NN[3,...] = np.roll(M1, 1, axis = 1) - M1
	delM1_NN[4,...] = np.roll(M1, (1,-1), axis = (0,1)) - M1
	delM1_NN[5,...] = np.roll(M1, (-1,1), axis = (0,1)) - M1

	if boundary == 1:
		delM1_NN[0,-1,...] = 0
		delM1_NN[1,0,...] = 0
		delM1_NN[2,:,-1,...] = 0
		delM1_NN[3,:,0,...] = 0
		delM1_NN[4,0,...] = 0
		delM1_NN[4,:,-1,...] = 0
		delM1_NN[5,-1,...] = 0
		delM1_NN[5,:,0,...] = 0

	# solve D1.g = delM1_NN for each (i,j)
	for ii in range(N1):
		for jj in range(N2):
			g[:,ii,jj,:] = np.linalg.lstsq(D1, delM1_NN[:,ii,jj,:], rcond=None)[0]

	delM1 = g[:2,...] # (a, i, j, s)

	M1_new = M1.copy()
	M1_new = M1_new + g[2,...]

	return M1_new, delM1




	# G1 = 2*math.pi*np.array([1,-1/math.sqrt(3)])/N0
	# G2 = 2*math.pi*np.array([0,2/math.sqrt(3)])/N0

	# Gmat = np.zeros([N0,N0,2])
	# Fmat = np.zeros([N0,N0,N0,N0], dtype = complex) # exp(-iG_ij*r_kl)

	# for ii in range(N0):
	# 	for jj in range(N0):
	# 		Gmat[ii,jj,:] = G1*ii+G2*jj # (i,j, a)
	# 		for kk in range(N0):
	# 			for ll in range(N0):
	# 				Fmat[ii,jj,kk,ll] = np.exp(- 1j*2*math.pi*(ii*kk+jj*ll)/N0) # (i,j,k,l)


	# C1 = np.zeros([N0,N0,3], dtype = complex)
	# C2 = np.zeros([N0,N0,3], dtype = complex)

	# C1 = np.einsum('ijkl, kls -> ijs', Fmat, M1)/N0**2
	# C2 = np.einsum('ijkl, kls -> ijs', Fmat, M2)/N0**2

	# # s: direction of M, a: direction of derivative, i,j: spatial index
	# delM1 = np.real(1j*np.einsum('ija, ijs, ijkl -> askl', Gmat, C1, np.conjugate(Fmat)) )# (a, s, i, j)
	# delM2 = np.real(1j*np.einsum('ija, ijs, ijkl -> askl', Gmat, C2, np.conjugate(Fmat)) )# (a, s, i, j)

	# plt.imshow(delM1[0,2,:,:])
	# plt.show()



#
# test for calculating skyrmion number with charge q
#
def test_skyrmion_in_trilattice(N0, q):


	a1 = np.array([1,0])
	a2 = np.array([1/2, math.sqrt(3)/2])
	
	a0 = 10 # cut-off = size of skyrmion core

	N0_h = int(N0/2) 
	r0 = N0_h*(a1+a2) # center of a skyrmion

	M0 = np.zeros([N0,N0,3])
	for ii in range(N0):
		for jj in range(N0):
			r_ij = ii*a1 + jj*a2 - r0
			x, y = r_ij
			R_ij = np.linalg.norm(r_ij) 
			alpha_ij = np.arctan2(y,x)
			M0[ii,jj,0] = 2*(R_ij * a0)**q * math.cos(q*alpha_ij)/(R_ij**(2*q) + a0**(2*q))
			M0[ii,jj,1] = -2*(R_ij * a0)**q * math.sin(q*alpha_ij)/(R_ij**(2*q) + a0**(2*q))
			M0[ii,jj,2] = (R_ij**(2*q) - a0**(2*q))/(R_ij**(2*q) + a0**(2*q))

	S1 = spin("random", N0)
	S1.M[0,:,:,0,:] = M0

	S1.plot_spins() 

	print(get_skyrmion_number(M0))

