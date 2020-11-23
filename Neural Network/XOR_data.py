#XOR data
#Ausra Saudargiene

import numpy as np
import matplotlib.pyplot as plt
import math

# fix random seed for reproducibility
np.random.seed(25)

#------------------------------------------------------
#Data generation
#------------------------------------------------------ 
   
def generate_data(N):
       
    N_half=round(N_train/2) #number of vectors in first class sublasses (red)
    #parameters for data generation
    #two dimensional Gaussian distribution  
    mean1a = np.array([1, 0])  #  mean vector class 1
    mean1b = np.array([0, 1])  #  mean vector class 1
    mean2a = np.array([0, 0])  #  mean vector class 2
    mean2b = np.array([1, 1])  #  mean vector class 2
    var1=0.1 # variance of x1 feature 
    var2=0.1 # variance of x2 feature 
    cor12=0.1 #correlation coefficient between x1 and x2
    cov12=cor12*math.sqrt(var1)*math.sqrt(var2) #covariance
    cov1 = np.array([[var1, cov12], [cov12, var2]])  #  covariance matrix class 1
    cov2 = cov1  #  covariance matrix class 2
    data1a = np.random.multivariate_normal(mean1a, cov1, N_half)
    data1b = np.random.multivariate_normal(mean1b, cov1, N_half)
    data2a= np.random.multivariate_normal(mean2a, cov2,  N_half)
    data2b= np.random.multivariate_normal(mean2b, cov2,  N_half)
    targets1=np.zeros((N, 1))
    targets2=np.ones((N, 1))
    data1=np.append(data1a,data1b,axis=0)
    data2=np.append(data2a,data2b, axis=0 )
         
    return  data1,  data2, targets1, targets2
    
 
#----------------------------------------------------------------------------    
if __name__ == '__main__':
    
    #number of vectors in each class
    N_train=200
    data1,  data2, x, y = generate_data(N_train)
        
    
    #--------------------------------------------------
    #Plot data
    # -------------------------------------------------
    plt.figure(figsize=(8, 6))
  
    fig = plt.figure(figsize=(6,4)) # indicating figure size
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(data1[:,0], data1[:,1], 'ro', data2[:,0], data2[:,1], 'bo')
    ax.set_title('XOR Data ', style='italic')
    ax.set_ylabel('x1', style='italic')
    ax.set_xlabel('x2', style='italic')
    plt.tight_layout()
    plt.show()
    fig.savefig('XORdata.png')   
        
      
