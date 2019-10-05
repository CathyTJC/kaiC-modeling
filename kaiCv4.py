#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 12:36:33 2019

@author: cathytang
"""
####ODE model simulation using KaiA, KaiB, KaiC system######
# avoid GPU due to Cholesky decompositions
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, DotProduct, WhiteKernel,ConstantKernel as C
import warnings
warnings.filterwarnings("ignore")

# Construct ODE systems
def dxdt_mm(X,t,K0,KA,C_kaiC,C_kaiA,m,K_half):
    '''The equations defining an enzyme-substrate reaction.
    
    This function contains three equations that return the change in
    concentration of the substrate, complex, and product with respect
    to time.
    
    Arguments:
        X is an ndarray of initial values of:
            [T] -- the concentration of a substrate.
            [D] -- the concentration of substrate 2
            [S] -- the concentration of substrate 3
        t is the current timepoint.
        Kxy are rate constants.
    '''
    T = X[0]
    D = X[1]
    S = X[2]
    
    #[Kut0,Ktd0,Ksd0,Kus0,Ktu0,Kdt0,Kds0,Ksu0] = K0
    #[KutA,KtdA,KsdA,KusA,KtuA,KdtA,KdsA,KsuA] = KA 
    
    
    A = max(0, C_kaiA-2*m*S)
    [Kut,Ktd,Ksd,Kus,Ktu,Kdt,Kds,Ksu] = K0 + np.true_divide(np.dot(KA,A),(K_half + A))
    U = C_kaiC -(T+D+S)
    T_new = Kut * U + Kdt * D - Ktu * T - Ktd * T
    D_new = Ktd * T + Ksd * S - Kdt * D - Kds * D
    S_new = Kus * U + Kds * D - Ksu * S - Ksd * S
        
    return T_new, D_new, S_new

# Create a (1000,) ndarray of time points from 0 to 100.
t = np.linspace(0, 100, 1000)

### Constants value
#rate constants (h-1): Basal rate (wtihout KaiA)
K0 =[0,0,0,0,0.21,0,0.31,0.11]

#maximal effect of KaiA
KA =[0.479077,0.212923,0.505692,0.0532308,0.0798462,0.1730000,-0.319385,-0.133077]

# IC (uM)
T0 = 0.68
D0 = 1.36
S0 = 0.34
IC = [T0, D0, S0]

#Other
K_half = 0.43
m1 = 1 #if KaiB exist
m2 =0 # if KaiB absent
C_kaiA = 1.3
C_kaiC = 3.4

# Call the ODE solver with the following arguments:
X_true, infodict = odeint(dxdt_mm, IC, t, args=(K0,KA,C_kaiC,C_kaiA,m1,K_half), full_output=True)
print(infodict['message'])

######plot Conc. vs time #########
X_percent = X_true
fig, ax = plt.subplots()
Total = np.sum(X_percent,axis =1)
U = C_kaiC-Total
tmp = ax.plot(t, X_true)
tmp = ax.legend(['T-KaiC', 'ST-KaiC', 'S-KaiC'])
xlabel = ax.set_xlabel('Time(h)')
ylabel = ax.set_ylabel('KaiC Conc')
plt.show()

################# Implement gaussian process regression
# adapted from sklearn example code

# Generate sample data
X = t[:,np.newaxis]
y = X_percent

# randomly choose data points as exp data points
x=np.random.randint(0, high=len(X), size=20, dtype='l')
X_sample = X[x]
y_sample=y[x]


######## New GP
# Instantiate a Gaussian Process model
kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))
gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9)

# Fit to data using Maximum Likelihood Estimation of the parameters
gp.fit(X_sample, y_sample)

# Make the prediction on the meshed x-axis (ask for MSE as well)
y_pred, sigma = gp.predict(X, return_std=True)

# Plot the function, the prediction and the 95% confidence interval based on
# the MSE
plt.figure()
plt.plot(X, y, '.', markersize=3, label=u'Observations')
plt.plot(X, y_pred, '-', label=u'Prediction')
for i in range(0,y.shape[1]):
    plt.fill(np.concatenate([X, X[::-1]]), np.concatenate([y_pred[:,i] - 1.9600 * sigma,
                             (y_pred[:,i] + 1.9600 * sigma)[::-1]]),
             alpha=.3, fc='b', ec='None', label='95% confidence interval')
plt.xlabel('$t$')
plt.ylabel('KaiC conc.')
plt.legend(loc = 'best')

#infer parameter
#The optimal parameter values can be ascertained by maximum likelihood estimation

# In[]
# rerun integration with values inffered from GP
K0_gp =[0,0,0,0,0.21,0,2.773095, 1.958643]

#maximal effect of KaiA
KA_gp =[0.479077,0.212923,0.505692,0.0532308,0.0798462,0.1730000,-0.319385,-0.133077]

# Call the ODE solver with the following arguments:
X_gp, infodict = odeint(dxdt_mm, IC, t, args=(K0_gp,KA_gp,C_kaiC,C_kaiA,m1,K_half), full_output=True)
print(infodict['message'])

######plot Conc. vs time #########

fig, ax = plt.subplots()
ax.plot(t, X_gp,'.', label=u'GP_Prediction')
tmp = ax.legend(['T-KaiC', 'ST-KaiC', 'S-KaiC'])
xlabel = ax.set_xlabel('Time(h)')
ylabel = ax.set_ylabel('% KaiC')
ax.set_title('GP')
plt.show()


############# rerun integration with values inffered from rnn############
x_rnn = np.load('result.npy')
param_RNN = np.load('infer_p.npy')
K0_rnn = param_RNN[0][0:8]
KA_rnn = param_RNN[0][8:16]

# Call the ODE solver with the following arguments:
X_rnn, infodict = odeint(dxdt_mm, IC, t, args=(K0_rnn,KA_rnn,C_kaiC,C_kaiA,m1,K_half), full_output=True)
print(infodict['message'])

######plot Conc. vs time #########

fig, ax = plt.subplots()
ax.plot(t, X_rnn,'.', label=u'GP_Prediction')
tmp = ax.legend(['T-KaiC', 'ST-KaiC', 'S-KaiC'])
xlabel = ax.set_xlabel('Time(h)')
ylabel = ax.set_ylabel('KaiC Conc')
ax.set_title('RNN')
plt.show()



# In[] ############### Infer parameter from linear regression
# selection 100 random time points and set them as observations
t_lr=np.random.choice(range(len(t)), 100, replace=False)
t_lr=np.sort(t_lr)
t_obser = [t[i] for i in t_lr]
x_obser = np.asarray([X_true[i] for i in t_lr])
U_obser = np.asarray([U[i] for i in t_lr])


# a fucntion that get the column of each multi matrix
def column(matrix, i):
    return [row[i] for row in matrix]

S_ob = column(x_obser, 2)
D_ob = column(x_obser, 1)
T_ob = column(x_obser, 0)


A_ob =[]
for num in range(0,len(S_ob)):
    A_ob.append( max(0,C_kaiA - 2 * S_ob[num]))

# change the sys to a linear reg (least squre) to get param
#now get the data count for polynomial terms
new_list = [x+K_half for x in A_ob]
aT =np.true_divide(np.dot(A_ob,U_obser), new_list)
aD =np.true_divide(np.dot(A_ob,D_ob), new_list)
aS =np.true_divide(np.dot(A_ob,S_ob), new_list)
aU=np.true_divide(np.dot(A_ob,U_obser), new_list)



# use least squre to solve for x =[a0 ,a1,a2,a3].T Ax = b
# combine each term to get matrix A 
A =d = np.zeros((2,16))
for i in range(0,len(aD)):
    X = np.array([[U_obser[i], aU[i], D_ob[i], aD[i],-T_ob[i], -aT[i], -T_ob[i], -aT[i],0,0,0,0,0,0,0,0],
                 [0,0,-D_ob[i], -aD[i],0,0,T_ob[i], aT[i], S_ob[i], aS[i], -D_ob[i], -aD[i],0,0,0,0],
                 [0,0,0,0,0,0,0,0,-S_ob[i], -aS[i],D_ob[i], aD[i],U_obser[i], aU[i],-S_ob[i], -aS[i]]])    
    A = np.array(np.vstack((A, X)))
 
x_train_lr = A[2:]

# will lose the last 3 row when approximater the derivartive 
A = x_train_lr[:-3]


dS =[]
dT =[]
dD =[]
# Get Y_train
 # y is (dT/dt got from interpreting the slope)
for i in range(0,len(S_ob)-1):
    dS.append((S_ob[i+1]- S_ob[i])/(t_obser[i+1]-t_obser[i]))
    dT.append((T_ob[i+1]- T_ob[i])/(t_obser[i+1]-t_obser[i]))   
    dD.append((D_ob[i+1]- D_ob[i])/(t_obser[i+1]-t_obser[i]))   

Y_train = pd.DataFrame(np.array([dT, dD,dS]),index=['T', 'D','S'])
Y_train=Y_train.T    
test = np.array(Y_train)
b = np.array(np.reshape(test,3*len(dS)))

######### perform least square Ax=b min|Ax-b|_2 to get parameters!
# A 147*8, b: 147*1 

x_lr= np.linalg.lstsq(A, b, rcond=None)
# x = solve(A, b)
param_lr = x_lr[0]
residuals_lr = x_lr[1]

# ODE integration with values inffered from lest square
k0_lr = [param_lr[0],param_lr[6],param_lr[8],param_lr[12],param_lr[4],param_lr[2],param_lr[10],param_lr[14]]
KA_lr = [param_lr[1],param_lr[7],param_lr[9],param_lr[13],param_lr[5],param_lr[3],param_lr[11],param_lr[15]]
# Call the ODE solver with the following arguments:
X_lr, infodict = odeint(dxdt_mm, IC, t, args=(k0_lr ,KA_lr,C_kaiC,C_kaiA,m1,K_half), full_output=True)
print(infodict['message'])

######plot Conc. vs time #########
# IC (uM)
T0 = 0.68
D0 = 1.36
S0 = 0.34
IC = [T0, D0, S0]

#Other
K_half = 0.43
m1 = 1 #if KaiB exist
m2 =0 # if KaiB absent
C_kaiA = 1.3
C_kaiC = 3.4


fig, ax = plt.subplots()
ax.plot(t, X_lr)
tmp = ax.legend(['T-KaiC', 'ST-KaiC', 'S-KaiC'])
xlabel = ax.set_xlabel('Time(h)')
ylabel = ax.set_ylabel('% KaiC')
ax.set_title('Linear Regression (least square with poly term)')
plt.show()

# In[] Plot of combination of true, gp, and rnn results
t_plot=np.random.choice(range(len(t)), 100, replace=False)
t_plot=np.sort(t_plot)
t_obser = [t[i] for i in t_plot]

x_true_plot = np.asarray([X_true[i] for i in t_plot])
x_gp_plot = np.asarray([X_gp[i] for i in t_plot])
x_rnn_plot = np.asarray([X_rnn[i] for i in t_plot])
x_lr_plot = np.asarray([X_lr[i] for i in t_plot])

fig, ax = plt.subplots()
plt.ylim([0,1.8])
for i in [0,1,2]:
    if i == 0:
        color ='b'
    if i == 1:
        color ='g'
    if i == 2:
        color ='r'

    ax.plot(t, X_true[:,i],color+'-')
    ax.plot(t_obser, x_gp_plot[:,i],color +'-.',markersize=5)
    ax.plot(t_obser, x_rnn_plot[:,i],color+'o',markersize=4)
    ax.plot(t_obser, x_lr_plot[:,i],color+'*',markersize=5)
ax.set_ylim =([0, 1.8])
xlabel = ax.set_xlabel('Time(h)',size=14)
ylabel = ax.set_ylabel('KaiC Conc (3 phospho-form)',size=14)

legend = ax.legend(['T-KaiC-True', 'T-KaiC-GP','T-KaiC-RNN','T-KaiC-LinReg','ST-KaiC-True', 'ST-KaiC-GP', 'ST-KaiC-RNN','ST-KaiC-LinReg','S-KaiC-True', 'S-KaiC-GP',  'S-KaiC-RNN','S-KaiC-LinReg'],loc='top right',prop={'size': 6})

plt.show()

# In[] Evaluation, select 30 time points of each of the 3 species, calculate MSE
# randomly select observation time points
t_point=np.random.randint(0, high=len(t), size=100, dtype='l')
x_true = np.asarray([X_true[i] for i in t_point]).ravel()
x_gp = np.asarray([X_gp[i] for i in t_point]).ravel()
x_rnn = np.asarray([X_rnn[i] for i in t_point]).ravel()
x_lr = np.asarray([X_lr[i] for i in t_point]).ravel()

mse_sum_lr = 0
mse_sum_gp = 0
mse_sum_rnn = 0
# calculater MSD of gp and rnn
for i in range(0,len(x_gp)):
  MSE_gp = (x_gp[i]-x_true[i])*(x_gp[i]-x_true[i])
  MSE_rnn = (x_rnn[i]-x_true[i])*(x_rnn[i]-x_true[i])
  MSE_lr = (x_lr[i]-x_true[i])*(x_lr[i]-x_true[i])
  mse_sum_gp += MSE_gp
  mse_sum_rnn += MSE_rnn
  mse_sum_lr += MSE_lr
  
mse_GP =mse_sum_gp/len(x_gp)
mse_RNN = mse_sum_rnn/len(x_rnn)
mse_lr = mse_sum_lr/len(x_lr)
print('The MSE for Gaussian Process is: ',mse_GP)
print('The MSE for Recurrent Neural Network is: ',mse_RNN)
print('The MSE for Linear Regression is: ',mse_lr)
