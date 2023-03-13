import numpy as np
from scipy.optimize import curve_fit
import sys
from mpl_toolkits.axes_grid1 import make_axes_locatable
import math
import matplotlib.pyplot as plt
from matplotlib import colors
import matplotlib.transforms as mtransforms

#
# plot the interlayer coupling
# 	
def save_fig_J(fitData, filename):

	divnorm=colors.TwoSlopeNorm(vmin=-1.5, vcenter=0., vmax=1.5)
	fig, ax = plt.subplots()

	transform = mtransforms.Affine2D().skew_deg(30, 0)
	im = ax.imshow(fitData, interpolation='none', origin='lower', \
					extent=[0, 1, 0, 1], clip_on=True, norm=divnorm, cmap='bwr')

	trans_data = transform + ax.transData
	im.set_transform(trans_data)
	ax.axis('off')
	# display intended extent of the image
	x1, x2, y1, y2 = im.get_extent()
	ax.plot([x1, x2, x2, x1, x1], [y1, y1, y2, y2, y1],	transform=trans_data, alpha = 0.0)

	ax.tick_params(labelsize = 4.0)
	divider = make_axes_locatable(ax)
	cax = divider.append_axes('right', size='5%', pad=0.05)
	fig.colorbar(im, cax=cax, orientation='vertical')
	cax.tick_params(labelsize = 4.0)

	plt.savefig(filename)

#
# read delE extracted from Nano. letter
#
def read_delE(file_name):
	d = np.genfromtxt(file_name, delimiter=',')
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
# fitting the data from Nano Letter by several harmonics preserving the C3 symmetry
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



########################################################################

N0 = int(sys.argv[1])
job_name_str = sys.argv[2]
fig_ratio = float(sys.argv[3])
harmonics = int(sys.argv[4])


assert fig_ratio >= 0 and fig_ratio <= 1 
assert harmonics > 1 and harmonics <= 9


#
# set up interlayer coupling 
#
ratio_delE_J = -4*(1.5**2)

points, z2 = read_delE('deltaE_Nano_2.csv')
points, z3 = read_delE('deltaE_Nano_3.csv')

z = fig_ratio*z2 + (1-fig_ratio)*z3



J0_perp = z[6]/ratio_delE_J # J_perp of AB' (monoclinic) stacking: AFM (J < 0)

popt, pcov, infodict, mesg, ier = curve_fit(J_fit, xdata=points.T, ydata=z, p0=[1]*(4*harmonics+5), full_output = True)


# 
# grid for interpolation 
# 
N_grid = 3*N0+1
r_grid = np.linspace(0,1, num = N_grid)
grid_x, grid_y = np.meshgrid(r_grid, r_grid)
r_grid = np.squeeze(np.array([[grid_x.reshape(N_grid**2,1)],[grid_y.reshape(N_grid**2,1)]]))
fitData=J_fit(r_grid, *popt).reshape(N_grid, N_grid)
# delE_interpolated = interpolate.griddata(points, z, (grid_x, grid_y), method='cubic').T # oddly the axis are swapped.

# divnorm=colors.TwoSlopeNorm(vmin=-15., vcenter=0., vmax=15)
# plt.imshow(fitData, norm=divnorm, cmap = 'bwr')
# plt.colorbar()
# plt.show()


J_perp_a = fitData[:-1:3,:-1:3]/ratio_delE_J
J_perp_b = fitData[1:-1:3,1:-1:3]/ratio_delE_J

np.savetxt(job_name_str + '_J_perp_a.dat', J_perp_a, fmt = '%10.6f', delimiter=',')
np.savetxt(job_name_str + '_J_perp_b.dat', J_perp_b, fmt = '%10.6f', delimiter=',')
with open(job_name_str + '_J0_perp.dat', 'w') as f:
    f.write(f'{J0_perp}')

# save_fig_J(J_perp_a, 'test.pdf')