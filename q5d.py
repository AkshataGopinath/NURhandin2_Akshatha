import numpy as np
import matplotlib.pyplot as plt

######## 5(d) #########

def fft(x):
    """
    FFT algorithm for 1D- using recursive Cooley-Turkey FFT
    Inputs:
    x: discrete function whose fft we want to find
    """
    N = len(x)
    if (N & (N - 1)) != 0: print("Array length must be a power of 2")
    
    if N > 2:
        # we are assuming N is a power of 2, so splitting into two parts
        x = np.concatenate((fft(x[::2]), fft(x[1::2])))

    for k in range(0, N//2):
        w = np.exp(-2*np.pi*k*1j/N)
        t = x[k]
        x[k] = t + w*x[k+ N//2]
        x[k+ N//2] = t - w*x[k+ N//2]

    return x



"""## Checking for 1D FFT function"""

# creating a 1d rectangular function
def box1d(x, width):
    """
    Inputs:
    x -  range in which function is defined
    width - widh of rectangular function
    """
    xx = np.empty(len(x))
    for i,x in enumerate(x):
         xx[i]=1 if abs(x)<= width//2 else 0
    return xx

# analytical Fourier transform of the rectangular function: sinc function
def box1d_ft(x):
    xx = np.zeros(len(x))
    for i,x in enumerate(x): 
        if x!=0: xx[i]= np.sin(x)/(x)
    return xx

def qn5d():    
	f1= box1d(np.arange(-64, 64, 1), 20)

	# analytical fourier transform of box_1d:
	f1_th = box1d_ft(np.arange(-64, 64, 1))
	# FFT using the numpy FFT function
	f1_np = np.fft.fft(np.array(f1, dtype=complex)).real
	# FFT using the function written above
	f1_me = fft(np.array(f1, dtype=complex)).real

	# input function: boxcar
	plt.figure(figsize=(10,5))
	plt.suptitle("1D FFT verification: it is found that the fourier transform found by the \n written fft function matches with the result of the numpy fft function and the analytical FT of boxcar")
	plt.subplot(121)
	plt.plot(f1)
	plt.title('Input 1D function')

	plt.subplot(122)
	plt.plot(f1_np, label='Numpy FFT')# linestyle='dashed')# marker='o')
	plt.plot(f1_me, linestyle='dashed', label='My FFT')
	plt.title('\n \n Fourier transform of 1D function')
	#plt.plot(np.fft.fftshift(20*f1_th), label='Analytical FFT')
	plt.legend()
	plt.tight_layout()

	plt.savefig('5d_fft1d.png')
	
if __name__ == '__main__':
	print('Question 5d:')
	qn5d()
