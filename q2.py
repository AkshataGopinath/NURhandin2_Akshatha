import numpy as np
import matplotlib.pyplot as plt
from q1b import box_muller

def k(kx, ky):
    return np.sqrt(kx**2 + ky**2)

def gen_2Dgauss(n,N):
    """
    Inputs:
    n : power of the power law spectrum the field is to follow
    N: number of gridpoints in one dimension, to generate an NxN grid in 2D

    """

    grid = np.zeros((N,N), dtype = complex)
    # to satisfy the symmetry condition: k = 2*pi/N [0,1,2....Nq, 1-Nq, 2-Nq.....,-2,-1]
    # where Nq is the Nyquist frequency. Nq= N/2
    Nq = N//2
    c = (2*np.pi/N)
    for ix in range(0, N):
        if ix<= Nq:
            kx = c*ix
        else:
            kx = c*(ix-N) # (i-N) to obtain [1-Nq, 2-Nq...,-1] from [n/2+1, N/2+2....,N-1]
            
        for iy in range(0, N):
            if iy<= Nq:
                ky = c*iy
            else:
                ky = c*(iy-N) # (i-N) to obtain [1-Nq, 2-Nq...,-1] from [n/2+1, N/2+2....,N-1]
            
            kxy = np.sqrt(kx**2 + ky**2)
            if kxy == 0.: continue
            # variance = k**n, so sigma=k**(n/2)
            kpn = kxy**(n/2)
            
            # using numpy gaussian random generator, since my function box_muller,
            # takes too long when executed 1024*1024 times and slows down the process. 
            
            #real = box_muller(0, kpn, 1)
            #imag = box_muller(0, kpn, 1)
            real = np.random.normal(0, kpn)
            imag = np.random.normal(0, kpn)
            grid[ix, iy] = real + 1j*imag
        
    #at k=0 and k=N/2, there is no imaginary part, in order to satisfy the symmetry condition
    
    # at (N/2, N/2), (0, N/2), (N/2,0), (0, 0), field is real valued
    grid[Nq, Nq] = box_muller(0.,k(Nq, Nq)**(n/2), 1) + 1j*0
    grid[Nq, 0] = box_muller(0.,k(Nq, 0)**(n/2), 1) + 1j*0
    grid[0, Nq] = box_muller(0.,k(0, Nq)**(n/2), 1) + 1j*0
    grid[0, 0] = 0. + 1j*0.
    
    # applying the symmetry condition to the columns at x = N/2, x=0; and row y=0
    for iy in range(Nq+1, N):
        grid[Nq, iy] = grid[Nq, N-iy].real - 1j*grid[Nq, N-iy].imag
        grid[iy, 0] = grid[N-iy, 0].real - 1j*grid[N-iy, 0].imag
        grid[0, iy] = grid[0, N-iy].real - 1j*grid[0, N-iy].imag

    # applying the symmetry condition to the rest of the grid  Y(-k) = conjugate(Y(k))
    for ix in range(Nq+1, N):
        for iy in range(1, N):
            grid[ix, iy] = grid[N-ix, N-iy].real - 1j*grid[N-ix, N-iy].imag
    return grid


def qn2():
	n1, n2, n3 = -1, -2, -3
	field1 = np.fft.ifft2(gen_2Dgauss(n1, 1024))*1024
	field2 = np.fft.ifft2(gen_2Dgauss(n2, 1024))*1024
	field3 = np.fft.ifft2(gen_2Dgauss(n3, 1024))*1024

	# n=-1
	print('The imaginary parts of all the fields is nearly zero, as required. \n The structures in the field are seen to grow in size as the power n goes from -1 to -3 ')

	plt.figure()
	plt.suptitle("Gaussian random field following power law with n=-1")
	plt.subplot(121)
	plt.imshow(field1.real, cmap='jet')
	plt.colorbar(fraction=0.046, pad=0.04)
	plt.title('Real part')


	plt.subplot(122)
	plt.imshow(field1.imag, cmap='jet')
	plt.colorbar(fraction=0.046, pad=0.04)
	plt.title('Imaginary part')
	plt.tight_layout()
	plt.savefig('GaussianRandomField1.png')


	# n=-2
	plt.figure()
	plt.suptitle("Gaussian random field following power law with n=-2")
	plt.subplot(121)
	plt.imshow(field2.real, cmap='jet')
	plt.title('Real part')
	plt.colorbar(fraction=0.046, pad=0.04)

	plt.subplot(122)
	plt.imshow(field2.imag, cmap='jet')
	plt.colorbar(fraction=0.046, pad=0.04)
	plt.title('Imaginary part')
	plt.tight_layout()
	plt.savefig('GaussianRandomField2.png')

	# n =-3
	plt.figure()
	plt.suptitle("Gaussian random field following power law with n=-3")
	plt.subplot(121)
	plt.imshow(field3.real, cmap='jet')
	plt.title('Real part')
	plt.colorbar(fraction=0.046, pad=0.04)

	plt.subplot(122)
	plt.imshow(field3.imag, cmap='jet')
	plt.colorbar(fraction=0.046, pad=0.04)
	plt.title('Imaginary part')
	plt.tight_layout()

	plt.savefig('GaussianRandomField3.png')
	
if __name__ == '__main__':
	print('Question 2:')
	qn2()
