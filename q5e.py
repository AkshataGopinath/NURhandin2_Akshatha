import numpy as np
import matplotlib.pyplot as plt
from q5d import fft

######## 5(e) #########
def fft2d(x2d):
    """
    2D FFT algorithm using fft() function written above
    Inputs:
    x: 2D discrete function(array) whose fft we want to find
    """
    X = np.empty(np.shape(x2d), dtype=complex)
    N1, N2 = X.shape

    for i1 in range(0, N1):
        X[i1,:] = fft(x2d[i1,:])
        
    for i2 in range(0, N2):
        X[:,i2] = fft(X[:,i2])

    return X

def fft3d(x3d):
    """
    3D FFT algorithm using fft2d() function written above
    Inputs:
    x: 3D discrete function(array) whose fft we want to find
    """
    X = np.empty(np.shape(x3d), dtype=complex)
    N1, N2, N3 = X.shape
    
    # 2D FFT in slices along x-axis, i.e. slices containing y-z plane
    for i1 in range(0, N1):
        X[i1,:, :] = fft2d(x3d[i1,:,:])
        
    # 1D FFT in for all columns in x-axis    
    for i2 in range(0,N2):
        for i3 in range(0, N3):
            X[:, i2, i3] = fft(X[:, i2, i3])

    return X

"""## Checking for 2D FFT function"""
def qn5e2d():
	# 2d boxcar array 
	input2d = np.full((20,20),1, dtype=complex)
	input2d = np.pad(input2d, ((22, 22), (22, 22)), 'constant', constant_values=(0,0))

	plt.figure(figsize=(10,5))
	plt.suptitle("2D FFT verification: it is found that the fourier transform found by the \n written fft function matches with the result of the numpy fft function. \n Analytically it should be a 2d sinc function, and it is seen that it resembles a 2D sinc function in the plots.")
	plt.subplot(131)
	plt.imshow(abs(input2d ))
	plt.title('Input boxcar array')


	plt.subplot(132)
	# FFT using the numpy FFT2 function
	plt.imshow(abs(np.fft.fftshift(fft2d(np.array(input2d)))))
	plt.title('\n \n Numpy FFT of boxcar array')

	# FFT using my implementation of fft2d
	plt.subplot(133)
	plt.imshow(np.fft.fftshift(abs(np.fft.fft2(np.array(input2d)))))
	plt.title('FFT of boxcar array found by \n my FFT implemetation')

	plt.savefig('5e_fft2d.png')

"""## Checking for 3D FFT function"""

# 3D symmetric gaussian function
def gauss3dsym(x,y,z, sigma):
    d = (x*x + y*y + z*z)
    g = (1/((2*np.pi)**(1.5)*sigma**3))*np.exp(-((d) /(2*sigma**2 ) ) )
    return g 

def qn5e3d():
	x, y, z = np.meshgrid(np.linspace(-1,1,64),np.linspace(-1,1,64),np.linspace(-1,1,64))

	# input 3d gaussian 
	gauss3d = gauss3dsym(x,y,z,0.1)
	# FFT of 3d gaussian using nympy.fft.fftn
	gauss_f_th = np.fft.fftn(np.array(gauss3d, dtype='complex'))
	# 3D FFT uding my implementation
	gauss_f= fft3d(np.array(gauss3d, dtype='complex'))

	plt.figure(figsize=(10,10))
	plt.suptitle("3D FFT verification: it is found that the fourier transform found by the \n written fft function matches with the result of the numpy fft function. \n Analytically it should be a 3d gaussian function, and it is seen that it resembles a 3D gaussian function in the plots.")

	### x-y ####
	plt.subplot(331)
	plt.imshow(((gauss3d[:,:,31])))
	plt.title('\n \n \n Input gaussian array: x-y slice \n')
	plt.tight_layout()
	plt.colorbar(fraction=0.046, pad=0.04)

	plt.subplot(332)
	# FFT using the numpy FFT3 function
	plt.imshow(np.fft.fftshift(abs(gauss_f_th[:,:,31])))
	plt.title('\n \n \n Numpy FFT: x-y slice \n')
	plt.tight_layout()
	plt.colorbar(fraction=0.046, pad=0.04)

	# FFT using my implementation of fft3d
	plt.subplot(333)
	plt.imshow(np.fft.fftshift(abs(gauss_f[:,:,31])))
	plt.title('\n \n \n FFT by my implemetation: x-y slice \n')
	plt.tight_layout()
	plt.colorbar(fraction=0.046, pad=0.04)


	### x-z ####
	plt.subplot(334)
	plt.imshow(((gauss3d[:,31,:])))
	plt.title('\n \n \n Input gaussian array: x-z slice \n')
	plt.tight_layout()
	plt.colorbar(fraction=0.046, pad=0.04)

	plt.subplot(335)
	# FFT using the numpy FFT3 function
	plt.imshow(np.fft.fftshift(abs(gauss_f_th[:,31,:])))
	plt.title('\n \n \n Numpy FFT: x-z slice \n')
	plt.tight_layout()
	plt.colorbar(fraction=0.046, pad=0.04)

	# FFT using my implementation of fft3d
	plt.subplot(336)
	plt.imshow(np.fft.fftshift(abs(gauss_f[:,31,:])))
	plt.title('\n \n \n FFT by my implemetation: x-z slice \n')
	plt.tight_layout()
	plt.colorbar(fraction=0.046, pad=0.04)


	### y - z ####
	plt.subplot(337)
	plt.imshow(((gauss3d[31,:,:])))
	plt.title('\n \n \n Input gaussian array: y-z slice \n')
	plt.tight_layout()
	plt.colorbar(fraction=0.046, pad=0.04)

	plt.subplot(338)
	# FFT using the numpy FFT3 function
	plt.imshow(np.fft.fftshift(abs(gauss_f_th[31,:,:])))
	plt.title('\n \n \n Numpy FFT: y-z slice \n')
	plt.tight_layout()
	plt.colorbar(fraction=0.046, pad=0.04)

	# FFT using my implementation of fft3d
	plt.subplot(339)
	plt.imshow(np.fft.fftshift(abs(gauss_f[31,:,:])))
	plt.title('\n \n \n FFT by my implemetation: y-z slice \n')
	plt.tight_layout()
	plt.colorbar(fraction=0.046, pad=0.04)


	plt.savefig('5e_fft3d.png')


if __name__ == '__main__':
	print('Question 5e:')
	print("""## Checking for 2D FFT function""")
	qn5e2d()
	print("""## Checking for 3D FFT function""")
	qn5e3d()
