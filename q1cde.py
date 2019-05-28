"""## 1c) , d) and e)"""

('######### 1 (c), (d) ##########')
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from astropy.stats import kuiper
from q1b import box_muller


# defining gaussian CDF function
def cdf_gauss(x, mu, sig):
    return 0.5*(1+ erf((x-mu)/(sig*np.sqrt(2))))

# using an approximation formula to calculate the error function erf
def erf(x1):
    a1, a2, a3, p = 0.3480242, -0.0958798, 0.7478556, 0.47047
    if x1>=0: 
        t= 1/(1 + p*x1)
        val = 1 - ((a1*t + a2*t**2 + a3*t**3)*np.exp(-x1**2))
    # since erf is an odd function
    elif x1<0: 
        x2 = -x1
        t= 1/(1 + p*x2)
        val = -(1 - ((a1*t + a2*t**2 + a3*t**3)*np.exp(-x2**2)))
        
    return val

def ks_significance(D, N):
    """
    function to determine the p-value from the KS test d-statistic
    """
    z =  (np.sqrt(N) + 0.12 + 0.11/np.sqrt(N))*D
    z2 = -2.0*z**2

    if z< 1.18:
        Pz = (np.sqrt(2*np.pi)/z) *((np.exp(-np.pi**2/(8*z**2)) + (np.exp(-np.pi**2/(8*z**2)))**9 + (np.exp(-np.pi**2/(8*z**2)))**25))

    elif z >= 1.18:
        Pz = 1.0 - 2*( np.exp(z2) - (np.exp(z2))**4 + (np.exp(z2))**9 )

    Qz = 1.0- Pz

    if Qz>1 : Qz= 0
    return Qz

def kuip_sig(D, N):
    """
    function to determine the p-value from the Kuiper test d-statistic
    """
    z =  (np.sqrt(N) + 0.155 + 0.24/np.sqrt(N))*D
    z2 = z**2
    Qkp =0
    for j in range(1, 10000):
        Qkp += (4*j**2*z2 - 1)* np.exp(-2*j**2*z2)

    return Qkp*2

def sort_quick(mylist, a, b): 
    pvt = mylist[b] #last element in the unsorted list is pivot
    i, j = a, b
    
    if a >= b: return
    
    while i<= j:
        # i goes up from 0 till mylist[i] > pvt
        while mylist[i] < pvt:
            i += 1
        # j goes down from b till mylist[j] < pvt
        while mylist[j] > pvt:
            j -= 1
        
        if i <= j:
            # when mylist[i] > pvt, or mylist[j] < pvt, interchange the elements at i and j positions, and continue
            mylist[i], mylist[j] = mylist[j], mylist[i]
            i += 1
            j -= 1
    # recursively apply the algorothm to the two sub arrays to the left and right of pivot    
    sort_quick(mylist, a, j)
    sort_quick(mylist, i, b)
    return mylist

def ks_test(data):
    """
    KS-test to determine if given data is gaussian distributed with mean 0 and variance 1
    (Kuiper's test also included here)
    """
    cdata = list(data)
    d = 0
    dkuip0=0
    dkuip1 =0
    cdf0 = 0
    cdf1 = 0 
    N = len(cdata)
    data1 = sort_quick(cdata, 0, N-1)
    for i,x in enumerate(data1):
        cdf0= i/N
        cdf1 = (i+1)/N
        cdf_model= cdf_gauss(x, 0, 1)
        d_new = max(abs(cdf_model - cdf0), abs(cdf_model - cdf1))
        dk0 = abs(cdf1 - cdf_model) 
        dk1 = abs(cdf_model - cdf0)
        # storing the maximum distance
        if d_new > d: d = d_new
        if dk0 > dkuip0: dkuip0 = dk0
        if dk1 > dkuip1: dkuip1 = dk1
            
    p_val = ks_significance(d , N)  
    d_kuiper = dkuip0 + dkuip1 
    p_kuiper = kuip_sig(d_kuiper, N)
    return d, p_val, d_kuiper, p_kuiper

## For use in Kuiper's test ##
# defining gaussian CDF function
def cdf_kuip(x, mu=0, sig=1):
    return 0.5*(1+ erf_kuip((x-mu)/(sig*np.sqrt(2))))

# using an approximation formula to calculate the error function erf
def erf_kuip(x1):
    a1, a2, a3, p = 0.3480242, -0.0958798, 0.7478556, 0.47047
    valss=[]
    for i in range(len(x1)):
        if x1[i]>=0: 
            t= 1/(1 + p*x1[i])
            val = 1 - ((a1*t + a2*t**2 + a3*t**3)*np.exp(-x1[i]**2))
        # since erf is an odd function
        elif x1[i]<0: 
            x2 = -x1[i]
            t= 1/(1 + p*x2)
            val = -(1 - ((a1*t + a2*t**2 + a3*t**3)*np.exp(-x2**2)))

        valss.append(val)
        
    return np.array(valss)

pvalue_th = np.empty(41)
pvalue= np.empty(41)
D_th= np.empty(41)
D = np.empty(41)
D_kuip = np.empty(41)
p_kuip = np.empty(41)
Dkuip_th = np.empty(41)
pkuip_th = np.empty(41)
#generating the input of 10^5 gaussian random numbers for ks-test
kstest_ip = box_muller(0,1,10**5) 
ks_range = np.logspace(1, 5, 41)

# performing ks-test and kuiper's test on given data
for j, ksr in enumerate(ks_range):
    # KS test using scipy.stats.kstest
    D_th[j] , pvalue_th[j] = stats.kstest( kstest_ip[0:int(ksr)], 'norm')
    #sort_dat = sort_quick(kstest_ip[0:int(ksr)], 0, len(kstest_ip[0:int(ksr)])-1)
    Dkuip_th[j], pkuip_th[j] = kuiper(kstest_ip[0:int(ksr)], cdf_kuip)
    # using my implementation of ks-test (Kuipers test included)
    D[j] , pvalue[j], D_kuip[j], p_kuip[j]=  ks_test(kstest_ip[0:int(ksr)])
ks_linrange = np.linspace(1, 5, 41)

def qn1c():
	plt.figure()
	plt.plot(ks_linrange, D, linewidth=2, label='calculated d-statistic')
	plt.plot(ks_linrange, D_th,linestyle='dashed',linewidth=1.5, label='scipy d-statistic')
	plt.title('KS test d-statistic: it is seen that as number of random numbers sampled increases, \n the sampled numbers are more and more gaussian in distribution')
	plt.xlabel('dex (Number of gaussian random numbers considered = $10^dex$')
	plt.ylabel('d-statistic')
	plt.legend()
	plt.savefig('ks_d.png', bbox_inches = "tight")

	plt.figure()
	plt.plot(ks_linrange, pvalue, linewidth=2, label='calculated p-value')
	plt.plot(ks_linrange, pvalue_th,linestyle='dashed',linewidth=1.5, label='scipy p-value')
	plt.title('KS test probability (p-value)')
	plt.xlabel('dex (Number of gaussian random numbers considered = $10^{dex})$')
	plt.ylabel('p-value')
	plt.legend()
	plt.savefig('ks_pval.png', bbox_inches = "tight")

	print("The calculated p-values for KS test match with the p-values given by scipy's ks-test function.")

def qn1d():
	plt.figure()
	plt.plot(ks_linrange, D_kuip, linewidth=2, label='calculated d-statistic')
	plt.plot(ks_linrange,Dkuip_th,linestyle='dashed',linewidth=1.5, label='scipy d-statistic')
	plt.title("Kuiper test d-statistic: the distance statistic calculated by me seems \n to be slightly higher than that calculated by astropy's kuiper's test, \n hence the p-values are slightly lower than expected.")
	plt.xlabel('dex (Number of gaussian random numbers considered = $10^dex$')
	plt.ylabel('d-statistic')
	plt.legend()
	plt.savefig('kuip_d.png', bbox_inches = "tight")

	plt.figure()
	plt.plot(ks_linrange, p_kuip, linewidth=2, label='calculated p-value')
	plt.plot(ks_linrange, pkuip_th,linestyle='dashed',linewidth=1.5, label='scipy p-value')
	plt.title('Kuiper test probability (p-value)')
	plt.xlabel('dex (Number of gaussian random numbers considered = $10^{dex})$')
	plt.ylabel('p-value')
	plt.legend()
	plt.savefig('kuiper_pval.png', bbox_inches = "tight")

('######### 1 e #########')

# using KS test to find the p-value for similarity between 2 discrete data sets.
def kstest_data(data1, data2):
    j1 = 0
    j2 = 0
    n1 = len(data1)
    n2 = len(data2)
    fn1 = 0.0
    fn2 = 0.0
    d = 0.0
    data11 = sort_quick(data1, 0, n1-1)
    data22 = sort_quick(data2, 0, n2-1)
    while (j1 < n1) and (j2 < n2):
        d1 = data11[j1]
        d2 = data22[j2]
        if d1 <= d2:
            j1 += 1
            fn1 = j1 / n1
            while (j1 < n1) and (d1 == data11[j1]):
                j1 += 1
                fn1 = j1 / n1
        if d2 <= d1:
            j2 += 1
            fn2 = j2 / n2
            while (j2 < n2) and (d2 == data22[j2]):
                j2 += 1
                fn2 = j2 / n2
        dt = np.abs(fn2 - fn1)
        if dt > d:
            d = dt
    en = ((n1 * n2) / (n1 + n2))
    prob = ks_significance(d, en)
    return d, prob

RN_file= np.loadtxt('randomnumbers.txt')

#finding the p-value using KS test between the generated gaussian random numbers \n and the 10 data sets provided
pval = np.empty(np.shape(RN_file)[1])
Dval= np.empty(np.shape(RN_file)[1])
pval_th = np.empty(np.shape(RN_file)[1])
Dval_th= np.empty(np.shape(RN_file)[1])
for j in range(np.shape(RN_file)[1]):
    Dval[j] , pval[j]=  kstest_data(RN_file[:, j], list(kstest_ip))
    Dval_th[j], pval_th[j] = stats.ks_2samp(RN_file[:, j], list(kstest_ip))

def qn1e():
	plt.figure()
	plt.plot(pval, linewidth=2, label='calculated p-value')
	plt.plot(pval_th, linestyle='dashed',linewidth=1.5, label='scipy p-value')
	plt.title('KS test probability (p-value)')
	plt.xlabel('Index of the dataset of random numbers')
	plt.ylabel('p-value')
	plt.legend()
	plt.savefig('ks_data.png', bbox_inches = "tight")
	print('KS test between the generated gaussian random numbers and the 10 datasets \n provided show that the p-value is high for dataset-3.\n So random number array 3 is most likely \n to be consistent with gussian random numbers with sigma=1 and mu=0')

	plt.figure()
	plt.plot(Dval, label='calculated d-statictic')
	plt.plot(Dval_th, linestyle='dashed',linewidth=1.5, label='scipy d-statistic')
	plt.ylabel('d-statistic')
	plt.xlabel('Index of the dataset of random numbers')
	plt.legend()
	plt.savefig('ksd_data.png', bbox_inches = "tight")
	print('Looking at the d-values, one can infer that dataset numbers 1, 4, 8 are very dissimilar to a gaussain distribution of mu=0 and sigma=1.')
	
	
if __name__ == '__main__':
	print('1 (c):')
	qn1c()
	print('1 (d):')
	qn1d()
	print('1 (e):')
	qn1e()
