import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import math 
from scipy.stats import pearsonr 
#plt.ioff()

def mse(y, a):
    return (y - a)**2

def d_mse(y, a):
    return -2 * (y - a)

#data X - features extracted from the patients voices
#data Y - motor detoriation progression and general detoriation progression
#rows - independnt samples, patients
X=np.loadtxt("X_Parkinson.dat")
Y=np.loadtxt("Y_Parkinson.dat")

N, n_features = X.shape
N, n_outputs = Y.shape

#output to be predicted
x=X[:,0]
y2=Y[:,0] # motor_UPDRS - Parkinson's disease, motor detoriation progression 
y1=Y[:,1] # total_UPDRS - Parkinson's disease, general detoriation progression 

mse=mean_squared_error(x, y2)
print('MSE: %.3f' % (mse))

rmse = math.sqrt(mean_squared_error(x, y2))
print('RMSE: %.3f' % (rmse))

mse=mean_squared_error(x, y1)
print('MSE for general detoriation progression : %.3f' % (mse))

rmse = math.sqrt(mean_squared_error(x, y1))
print('RMSE for general detoriation progression: %.3f' % (rmse))

correlation = pearsonr(x.ravel(), y2.ravel())
print('Pearsons correlation: %.3f, p value %.3f' % (correlation[0], correlation[1]))

correlation = pearsonr(x.ravel(), y1.ravel())
print('Pearsons correlation for general detoriation: %.3f, p value %.3f' % (correlation[0], correlation[1]))


#Plotting prediction and outputs
lw=2 # Plot linewidth.
plt.figure(1)
plt.plot(X[:,2], X[:,3], 'bo',lw)
plt.xlabel('Pattern 1')
plt.ylabel('Pattern 2')
plt.title('Data')
plt.savefig('Fig_data.png')
plt.show()


print('Data: patients ill with Parkinson Disease')
print('Number of input patients: %.3d' % N)
print('Number of input features: %.2d' % n_features)


