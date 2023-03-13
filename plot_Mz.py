import numpy as np
from numpy.random import default_rng
rng = default_rng()
import os
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from math import *
import sys
import glob
from PyPDF2 import PdfFileMerger, PdfFileReader
from natsort import natsorted
import matplotlib.transforms as mtransforms
from opt_einsum import contract
import f90nml

B_unit = 7.9291


def affine_plot(ax, Z, transform):
    im = ax.imshow(Z, interpolation='none',
                   origin='lower',
                   extent=[0, 1, 0, 1], clip_on=True)

    trans_data = transform + ax.transData
    im.set_transform(trans_data)

    # display intended extent of the image
    x1, x2, y1, y2 = im.get_extent()
    ax.plot([x1, x2, x2, x1, x1], [y1, y1, y2, y2, y1], "y--",
            transform=trans_data)
    ax.set_xlim(-5, 5)
    ax.set_ylim(-4, 4)


def get_R(Mz):
	R_AFM = 233
	R_FM = 90

	M_diff = abs(np.diff(Mz, axis = 0))
	R_3d_grid = R_FM + (R_AFM-R_FM)*M_diff/3.0
	R_2d_grid = np.sum(R_3d_grid, axis = 0)
	R_inv = 1/R_2d_grid
	R_total = 1/(np.concatenate(R_inv).sum())
	return R_total

def get_R_tunneling(M_3d):
	V0 = 0.18 # constant offset in potential eV
	J0 = 0.068 # magnetic splitting in eV
	d = 0.661 # thickness of each layer (nm)

	alpha = 5.12317 # sqrt(2*m/\hbar^2) in the WKB formula in 1/(nm*sqrt(eV))

	[Mx_n, My_n, Mz_n] = M_3d # (4, N0, N0)
	Mx_n = Mx_n/1.5
	My_n = My_n/1.5
	Mz_n = Mz_n/1.5

	T0 = 2.12997;

	Tx_up = T0*np.exp(-2*d*alpha*np.sqrt(V0-J0*Mx_n))
	Tx_dn = T0*np.exp(-2*d*alpha*np.sqrt(V0+J0*Mx_n))

	Ty_up = T0*np.exp(-2*d*alpha*np.sqrt(V0-J0*My_n))
	Ty_dn = T0*np.exp(-2*d*alpha*np.sqrt(V0+J0*My_n))
	
	Tz_up = T0*np.exp(-2*d*alpha*np.sqrt(V0-J0*Mz_n))
	Tz_dn = T0*np.exp(-2*d*alpha*np.sqrt(V0+J0*Mz_n))

	Gx = np.sum(np.prod(Tx_up, axis = 0) + np.prod(Tx_dn, axis = 0));
	Gy = np.sum(np.prod(Ty_up, axis = 0) + np.prod(Ty_dn, axis = 0));
	Gz = np.sum(np.prod(Tz_up, axis = 0) + np.prod(Tz_dn, axis = 0));

	G_total = Gx + Gy + Gz

	[Rx, Ry, Rz ]= [1/Gx, 1/Gy, 1/Gz]

	R_total = 1/G_total

	return np.array([Rx, Ry, Rz, R_total])



def get_R_tunneling_DM(Mz_3d, theta_3d):

	alpha_3d = theta_3d
	beta_3d = np.zeros((4,N0,N0))
	for ii in range(4):
		beta_3d[ii,:,:] = np.arccos(Mz_3d[ii,:,:]/muS_list[ii]) 
	
	U = np.zeros((2,2,4,N0,N0),dtype='complex')



	U[0,0,::] = np.cos(beta_3d/2)*np.exp(1j*alpha_3d/2);
	U[0,1,::] = -np.sin(beta_3d/2)*np.exp(1j*alpha_3d/2);
	U[1,0,::] = np.sin(beta_3d/2)*np.exp(-1j*alpha_3d/2);
	U[1,1,::] = np.cos(beta_3d/2)*np.exp(-1j*alpha_3d/2);

	T_up = 0.245013
	T_dn = 0.109081
	T0 = np.array([[T_up,0],[0,T_dn]])

	# Note the lk order for the transpose
	T1 = contract('ijpq,jk,lkpq->ilpq', U[:,:,0,::], T0, np.conj(U[:,:,0,::]))
	T2 = contract('ijpq,jk,lkpq->ilpq', U[:,:,1,::], T0, np.conj(U[:,:,1,::]))
	T3 = contract('ijpq,jk,lkpq->ilpq', U[:,:,2,::], T0, np.conj(U[:,:,2,::]))
	T4 = contract('ijpq,jk,lkpq->ilpq', U[:,:,3,::], T0, np.conj(U[:,:,3,::]))

	G_2d = np.abs(contract('ijpq, jkpq, klpq, lipq -> pq', T1, T2, T3, T4))

	G_total = contract('ij -> ', G_2d)

	R_total = 1/G_total

	return R_total


def get_R_tunneling_DM2(M_3d):
	T0_up = 0.245013
	T0_dn = 0.109081

	M_3d_in = np.asarray(M_3d) # (3, 4, N0, N0)
	M_3d_norm = np.linalg.norm(M_3d_in, axis = 0)
	M_3d_n = M_3d_in/M_3d_norm
		
	cosbeta = np.zeros((3,N0,N0))

	cosbeta[0,:,:] = contract('aij,aij->ij', M_3d_n[:,0,:,:], M_3d_n[:,1,:,:])
	cosbeta[1,:,:] = contract('aij,aij->ij', M_3d_n[:,1,:,:], M_3d_n[:,2,:,:])
	cosbeta[2,:,:] = contract('aij,aij->ij', M_3d_n[:,2,:,:], M_3d_n[:,3,:,:])

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

	return R_total	


	

dt_string = sys.argv[1]
input_file = f90nml.read(dt_string + '.nml')
if 'dmuS' in input_file['model_para']:
	dmuS = input_file['model_para']['dmuS']
else:
	dmuS = 0.0
muS = 1.5
muS_list = np.array([muS + dmuS, muS, muS, muS + dmuS]) 




# print(obs_data[:,2])

data_z = glob.glob(dt_string +'*_z.txt')
data_r = glob.glob(dt_string +'*_r.txt')

data_z = natsorted(data_z)
data_r = natsorted(data_r)

N_fig = len(data_z)

RR=np.empty(N_fig)
RT=np.empty((4, N_fig))
RD=np.empty((2, N_fig))

Mx_av = np.empty(N_fig)
My_av = np.empty(N_fig)
Mz_av = np.empty(N_fig)

obs_file = dt_string + '.txt'
if os.path.isfile(obs_file):
	obs_data = np.genfromtxt(obs_file)
	if os.stat(dt_string + '.txt').st_size > 0:
		tt = obs_data[:,0]
		Bz = obs_data[:,2]
		Mz_av = obs_data[:,-1]
		energy = obs_data[:,1]

		#
		# plotting the observables
		#
		fig, ax = plt.subplots(nrows = 3)
		ax[0].plot(tt,Bz, label=r"$B_z$")
		ax[1].plot(tt,Mz_av, label=r"$M_z$")
		ax[2].plot(tt,energy, label="energy")
		ax[0].set_xlabel('time')
		ax[1].set_xlabel('time')
		ax[2].set_xlabel('time')
		ax[0].set_ylabel(r'$B_z$')
		ax[1].set_ylabel(r'$M_z$')
		ax[2].set_ylabel('energy')
		plt.savefig(dt_string+'_t.pdf')
	else:
		B1 = input_file['model_para']['B_start']
		B2 = input_file['model_para']['B_final']
		N_B = input_file['model_para']['N_B']
		N_sweep = input_file['model_para']['N_sweep']
		B_sweep = B1 + (B2 - B1)*np.linspace(0,N_B, num=N_B+1)/N_B
		Bz = B_sweep

		for ii in range(2, N_sweep+1):
			if ii%2 == 0:
				Bz = np.append(Bz, np.flip(B_sweep))
			else:
				Bz = np.append(Bz, B_sweep)


for ii in range(N_fig):

	Mz_1d = np.genfromtxt(data_z[ii])
	theta = np.genfromtxt(data_r[ii])

	N0 = np.shape(Mz_1d)[1]

	Mz_3d = Mz_1d.reshape((4,N0,N0))
	theta_3d = theta.reshape((4,N0,N0))
	Mx_3d = np.zeros_like(Mz_3d)
	My_3d = np.zeros_like(Mz_3d)

	for ll in range(4):
		Mx_3d[ll,...] = np.sqrt(muS_list[ll]**2 - Mz_3d[ll,...]**2)*np.cos(theta_3d[ll,...])
		My_3d[ll,...] = np.sqrt(muS_list[ll]**2 - Mz_3d[ll,...]**2)*np.sin(theta_3d[ll,...])

	M_3d = [Mx_3d, My_3d, Mz_3d]

	
	Mx_av[ii] = np.mean(Mx_3d)
	My_av[ii] = np.mean(My_3d)
	Mz_av[ii] = np.mean(Mz_3d)

	fig = plt.figure(figsize=(8, 8))
	spec = fig.add_gridspec(4, 4)

	for spin_axis in range(3):
		for cols in range(4):
			ax = fig.add_subplot(spec[spin_axis+1,cols])
			transform = mtransforms.Affine2D().skew_deg(30, 0)
			# im = ax.imshow(Ma[cols*N0:(cols+1)*N0,:].T, interpolation='none', origin='lower', \
			#            extent=[0, 1, 0, 1], clip_on=True, vmin = -1, vmax = 1, cmap='bwr')
			im = ax.imshow(M_3d[spin_axis][cols,:,:].T, interpolation='none', origin='lower', \
					extent=[0, 1, 0, 1], clip_on=True, vmin = -1.6, vmax = 1.6, cmap='bwr')

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


	ax0 = fig.add_subplot(spec[0,1:3])
	ax0.plot(Bz*B_unit,Mz_av, label = 'Mz')
	ax0.set_xlabel(r'$B_z(T)$')
	ax0.set_ylabel(r'$M$')
	ax0.set_ylim([-1.6,1.6])
	ax0.scatter([Bz[ii]*B_unit], [Mx_av[ii]], marker='o')
	ax0.scatter([Bz[ii]*B_unit], [My_av[ii]], marker='P')
	ax0.scatter([Bz[ii]*B_unit], [Mz_av[ii]], marker='*')


	B0 = Bz[ii]*B_unit
	fig.suptitle(f"B0={B0}(T)")
	dataname = data_z[ii]
	figname = dataname[:-6] + '_a.pdf'
	plt.savefig(figname)
	plt.close()

	RR[ii] = get_R(Mz_3d)*N0*N0
	RT[:,ii] = get_R_tunneling(M_3d)*N0*N0
	RD[0, ii] = get_R_tunneling_DM(Mz_3d, theta_3d)*N0*N0
	RD[1, ii] = get_R_tunneling_DM2(M_3d)*N0*N0



#
# plotting B-M curve
#
fig, ax = plt.subplots()
ax.plot(Bz*B_unit,Mx_av, label = 'Mx')
ax.plot(Bz*B_unit,My_av, label = 'My')
ax.plot(Bz*B_unit,Mz_av, label = 'Mz')

ax.set_xlabel(r'$B_z(T)$')
ax.set_ylabel(r'$M$')
ax.set_ylim([-1.6,1.6])
ax.legend()
plt.savefig(dt_string+'_B-M.pdf')



#
# plotting B-R curve
#
import f90nml
nml = f90nml.read(dt_string + '.nml')
N_Bm = int(nml['model_para']['n_B'])
N_B = N_Bm + 1;

fig, ax = plt.subplots()
ax.plot(Bz[:N_B]*B_unit,RR[:N_B], c='k')
ax.plot(Bz[N_B:2*N_B]*B_unit,RR[N_B:2*N_B], c='r')
ax.set_xlabel(r'$B_z(T)$')
ax.set_ylabel(r'$R$')
ax.set_ylim([0,700])
plt.savefig(dt_string+'_B_RR.pdf')



fig, ax = plt.subplots()
# ax.plot(Bz[:N_B],RD[0,:N_B], 'k--')
# ax.plot(Bz[N_B:2*N_B],RD[0,N_B:2*N_B], 'r--')
ax.plot(Bz[:N_B]*B_unit,RD[1,:N_B], c='k', lw = 1)
ax.plot(Bz[N_B:2*N_B]*B_unit,RD[1,N_B:2*N_B], c='r', lw=1)
ax.set_xlabel(r'$B_z(T)$')
ax.set_ylabel(r'$R$')
ax.set_ylim([0,700])
plt.savefig(dt_string+'_B_RD.pdf')


fig, ax = plt.subplots(nrows=2, ncols=2)
ax[0,0].plot(Bz[:N_B]*B_unit,RT[-1,:N_B], c='k')
ax[0,0].plot(Bz[N_B:2*N_B]*B_unit,RT[-1,N_B:2*N_B], c='r')
ax[0,0].set_xlabel(r'$B_z(T)$')
ax[0,0].set_ylabel(r'$R_t$')

ax[0,1].plot(Bz[:N_B]*B_unit,RT[0,:N_B], c='k')
ax[0,1].plot(Bz[N_B:2*N_B]*B_unit,RT[0,N_B:2*N_B], c='r')
ax[0,1].set_xlabel(r'$B_z(T)$')
ax[0,1].set_ylabel(r'$R_x$')

ax[1,0].plot(Bz[:N_B]*B_unit,RT[1,:N_B], c='k')
ax[1,0].plot(Bz[N_B:2*N_B]*B_unit,RT[1,N_B:2*N_B], c='r')
ax[1,0].set_xlabel(r'$B_z(T)$')
ax[1,0].set_ylabel(r'$R_y$')

ax[1,1].plot(Bz[:N_B]*B_unit,RT[2,:N_B], c='k')
ax[1,1].plot(Bz[N_B:2*N_B]*B_unit,RT[2,N_B:2*N_B], c='r')
ax[1,1].set_xlabel(r'$B_z(T)$')
ax[1,1].set_ylabel(r'$R_z$')

plt.savefig(dt_string+'_B_RT.pdf')



pdf_list = glob.glob(dt_string + '*_a.pdf')

#sort the pdfs by file name:
pdf_list = natsorted(pdf_list)

# Call the PdfFileMerger
mergedObject = PdfFileMerger()

# Loop through all of the single pdfs and append them into one document
for file in pdf_list:
	mergedObject.append(PdfFileReader(file, 'rb'))
	os.remove(file)

# Write all the files into a file which is named as shown below
mergedObject.write(dt_string + '_f.pdf')


files = glob.glob(dt_string +'*')
new_folder = dt_string + '_file'
folder_index = 0
while os.path.isdir(new_folder) == True:
	new_folder = dt_string + str(folder_index) + '_file'
	folder_index += 1

os.mkdir(new_folder)
import shutil

for f in files:
	if os.path.isdir(f) == False:
		shutil.move(f, new_folder)