"""## 1b)"""

print('######### 1 (b) ##########')
import numpy as np
import matplotlib.pyplot as plt
from q1a import random

def box_muller(mu, sigma, n):
    """ 
    Box Mueller Transform to generate normally distributed random numbers 
    Inputs-
    mu : mean of required gaussian distribution that you want to sample 
    the random numbers from
    sigma : variance of the gaussian distribution
    n : number of random samples
    """
    u1 = random(0, 1, n)
    u2 = random(0, 1, n)
    rn_norm = sigma*(np.sqrt(-2*np.log(u1)) * np.cos(2*np.pi*u2)) + mu
    return rn_norm

mu1= 3
sig1= 2.7
# generating gaussian random numbers with mu1 and sig1
rn_norm = box_muller(mu1, sig1, 1000)

# to generate a theoretical gausian function
def gauss(x, mu, sigma): 
	g = (1/(np.sqrt(2*np.pi*sigma**2)))*np.exp(-((x-mu)**2/( 2 * sigma**2 ) ) )
	return g

g_range = np.linspace(mu1-5*sig1,mu1+5*sig1,100)
gaus = gauss(g_range, mu1, sig1)

def qn1b():
	plt.figure()
	plt.hist(rn_norm, bins= np.linspace(mu1-5*sig1,mu1+5*sig1,21), edgecolor='grey', linewidth=1.5, density=True, label='Histogram of random numbers')
	plt.title('Histogram of 1000 gaussian-distributed random numbers in (0,1) in bins of width 0.05: \n ')
	plt.xlabel('bins')
	plt.ylabel('frequency')

	#plt.figure()
	plt.plot(g_range,gaus, linewidth=1.5, color='black', label='theoretical gaussian')
	sigma_indices = np.arange(mu1-4*sig1,mu1+5*sig1-1, sig1)
	#sig_labels = ['-4\sigma', '-3\sigma','-2\sigma','-1\sigma','+1\sigma','+2\sigma','+3\sigma','+4\sigma',]
	plt.vlines([mu1 -sig1, mu1+sig1], 0, 0.2, color='red', linestyles='dashed', label='1 sigma')
	plt.vlines([mu1 -2*sig1, mu1+2*sig1], 0, 0.2, color='green', linestyles='dashed', label='2 sigma')
	plt.vlines([mu1 -3*sig1, mu1+3*sig1], 0, 0.2, color='orange', linestyles='dashed', label='3 sigma')
	plt.vlines([mu1 -4*sig1, mu1+4*sig1], 0, 0.2, color='blue', linestyles='dashed', label='4 sigma')
	plt.legend(loc='upper center', bbox_to_anchor=(1, 1), ncol=1, fancybox=True, shadow=True)

	plt.savefig('gaussRN.png', bbox_inches = "tight")
	
if __name__ == '__main__':
	print('1 (b):')
	qn1b()
