

import numpy as np
import matplotlib.pyplot as plt
import math





class Perceptron():
    def __init__(self,inputs, targets):
        self.inputs=inputs
        self.targets=targets
        [N,p]=self.inputs.shape 
      
        self.weights = np.random.rand(p+1,1)

    def sigmoid(self, x):
       
        return (1 / (1 + np.exp(-x)))

    def sigmoid_deriv(self, x):
       
        return np.exp(-x)/((1 + np.exp(-x))**2)

    def train(self, its, niu):
    
        [N,p]=self.inputs.shape 
        
        delta_weights = np.zeros((p+1,N))
  
        self.mse=np.zeros(its)
       
        self.T=list(range(its)) 
        
        for iteration in (range(its)):
            
            one_column=np.ones((N,1))
            inputs1=np.hstack((one_column, self.inputs)) 
            
           
            z = np.dot(inputs1, self.weights)
            
          
            activation = self.sigmoid(z)

           
            self.mse[iteration] =sum((activation - self.targets)**2)/N 
      
           
            for i in range(N): 
                cost_prime = 2*(activation[i] - self.targets[i])
                
                for j in range(p+1): 
                    delta_weights[j][i] = cost_prime * inputs1[i][j] * self.sigmoid_deriv(z[i])
                    
                    
            
            delta_avg = np.array([np.average(delta_weights, axis=1)]).T
            
           
            self.weights = self.weights - niu * delta_avg
         
            
    def output(self, inputs):
        inputs1=np.append([1], inputs) 
        return self.sigmoid(np.dot(inputs1, self.weights))
    
    
    def errors(self, data_test, targets_test):
       
        output_test = []
        error_test = 0
        
        N_test=len(data_test)
        
        for i in range(N_test):
           
            x_test = data_test[i]
            
            out = self.output(x_test)
            
            output_test.append(out.tolist())
            
            class_test=np.ravel(np.rint(output_test))
           
            class_test=class_test.astype(int)
            
            targets_test_1D=np.ravel(targets_test)
           
            if abs (class_test[i]-targets_test_1D[i]) > 0.01:
               
                    error_test+= 1
            
        error_test_percent= error_test/N_test*100         
        #print results
        print("Class observed")    
        print(targets_test_1D) 
        print("Class predicted")
        print(class_test) 
        return error_test, error_test_percent 
               
    
    
    def plot_mse(self):
        fig = plt.figure(figsize=(6,4))
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(self.T, self.mse, c='k')
        ax.set_title('Mean square error mse', style='italic')
        ax.set_ylabel('mse', style='italic')
        ax.set_xlabel('iterations', style='italic')
        plt.tight_layout()
        plt.show()
        fig.savefig('perceptron_mse.png')
        
    def plot_data(self, data1, data2):
        
        x1_min=min(np.append(data_train1[:,0],data2[:,0] ))
        x1_max=max(np.append(data_train1[:,0],data2[:,0] ))
        #calculating the corresponding x2 using the weights 
        x2_x1_min=(-self.weights[0]-self.weights[1]*x1_min)/self.weights[2]
        x2_x1_max=(-self.weights[0]-self.weights[1]*x1_max)/self.weights[2]
        #min and max x2
        x2_min=min(np.append(data1[:,1],data2[:,1] ))
        x2_max=max(np.append(data1[:,1],data2[:,1] ))
        
     
        fig = plt.figure(figsize=(6,4)) 
        ax = fig.add_subplot(1, 1, 1)
        
        ax.plot(data1[:,0], data1[:,1], 'ro', data2[:,0], data2[:,1], 'bo')
        ax.plot([x1_min, x1_max],  [x2_x1_min, x2_x1_max], 'k')
        ax.set_ylim([x2_min,x2_max])
        ax.set_title('Data and decision boundary', style='italic')
        ax.set_ylabel('x1', style='italic')
        ax.set_xlabel('x2', style='italic')
        plt.tight_layout()
        plt.show()
        fig.savefig('perceptron_data.png')



niu = 0.6 
epochs = 5

N_train=50 
N_test=50 

 
mean1 = np.array([0.8, 1])  
mean2 = np.array([0.5, 0])   
var1=1 
var2=2
var3=5
cor12=0.8 



cov12=cor12*math.sqrt(var1)*math.sqrt(var2)*math.sqrt(var3) 
cov1 = np.array([[var1, cov12], [cov12, var2]])  
cov2 = cov1  
data_train1 = np.random.multivariate_normal(mean1, cov1, N_train)
data_train2 = np.random.multivariate_normal(mean2, cov2, N_train)
targets_train1=np.zeros((N_train, 1))
targets_train2=np.ones((N_train, 1))
data_train=np.append(data_train1,data_train2, axis=0 )
targets_train=np.append(targets_train1,targets_train2, axis=0 )

data_test1 = np.random.multivariate_normal(mean1, cov1, N_test)
data_test2 = np.random.multivariate_normal(mean2, cov2, N_test)
targets_test1=np.zeros((N_test, 1))
targets_test2=np.ones((N_test, 1))
data_test=np.append(data_test1,data_test2, axis=0 )
targets_test=np.append(targets_test1,targets_test2, axis=0 )


mdiff=mean1-mean2

inv_cov1=np.linalg.inv(cov1)

m0=np.dot(mdiff,inv_cov1)
mahalanobis_distance_square=np.dot(m0,mdiff)
mahalanobis_distance=math.sqrt(mahalanobis_distance_square)



perceptron = Perceptron(data_train, targets_train) 








mse=perceptron.train(epochs, niu) 

[error_train, error_train_percent]=perceptron.errors(data_train, targets_train)
print("Training errors: %3d" % (error_train))
print("Training errors: %4.1f %%" % (error_train_percent))
#testing errors
[error_test, error_test_percent]=perceptron.errors(data_test, targets_test)
print("Test errors: %3d" % (error_test))
print("Test errors: %4.1f %%" % (error_test_percent))


#plot mse and data
perceptron.plot_mse() 
perceptron.plot_data(data_train1, data_train2) 
perceptron.plot_data(data_test1, data_test2) 
print("Mahalanobis distance between classes: %4.1f " % (mahalanobis_distance))


# In[ ]:




