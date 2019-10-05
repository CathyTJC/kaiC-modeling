# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 15:14:56 2019

@author: zhong
"""

#! -*- coding: utf-8- -*-




# In[] Data generation

def vectorfield(w,t, p):
    """
    Defines the differential equations for the chemical reaction system.
    Arguments:
        w :  vector of the state variables: w = [y1,y2,y3]
        t :  time (not necessary)
        p :  vector of the parameters:  p 
    """
    T,D,S = w                       # vector space                      
    K0=p[0:8]          # parameter space
    KA=p[8:16]
    C_kaiC=p[16]
    C_kaiA=p[17]
    m=p[18]
    K_half=p[19]


    A = max(0, C_kaiA-2*m*S)
    #A =  C_kaiA-2*m*S
    
    [Kut,Ktd,Ksd,Kus,Ktu,Kdt,Kds,Ksu] = K0 + np.true_divide(np.dot(KA,A),(K_half + A))
    U = C_kaiC -(T+D+S)
    func1 = Kut * U + Kdt * D - Ktu * T - Ktd * T
    func2 = Ktd * T + Ksd * S - Kdt * D - Kds * D
    func3 = Kus * U + Kds * D - Ksu * S - Ksd * S

    # Create f = (y1,y2,y3):
    dydt = [func1,func2,func3]
    #print(dydt)
    return dydt

# In[] ODE solver

from scipy.integrate import odeint
import numpy as np
# ODE solver parameters
abserr = 1.0e-8
relerr = 1.0e-6
#stoptime = 50
#numpoints = 1000

# Create the time samples for the output of the ODE solver.
t = np.linspace(0, 100, 1000)
#t=np.asarray(t)

# Pack up the parameters 

 
K0 =[0,0,0,0,0.21,0,0.31,0.11]
KA =[0.479077,0.212923,0.505692,0.0532308,0.0798462,0.1730000,-0.319385,-0.133077] 
C_kaiC = 3.4
C_kaiA = 1.3
m = 1
K_half = 0.43



p=K0+KA+[C_kaiC]+[C_kaiA]+[m]+[K_half]

np.save('true_p.npy',p)
#initial conditions
w0 = [0.68, 1.36,0.34]
# In[]
# Call the ODE solver.
wsol = odeint(vectorfield, w0, t, args=(p,),
              atol=abserr, rtol=relerr)
np.save('true_solution.npy',wsol)
# In[] randomly choose data points as exp data points


x=np.random.randint(0, high=len(t), size=15, dtype='l')
x=np.sort(np.append(x, 0))
wsol_list=np.array(np.round(wsol,4)).tolist()

exp_t=[ t[i] for i in x ]
exp_t=[round(x,4) for x in exp_t]

exp_sol=[ wsol_list[i] for i in x ]


exp=dict( zip( exp_t, exp_sol))
 
#del x, exp_t, exp_sol, wsol_list
np.save('observations',exp)

# In[] Visualization

from matplotlib import pyplot as plt

plt.figure()

plt.plot(t, wsol[:,0])
plt.plot(t,wsol[:,1])
plt.plot(t,wsol[:,2])
plt.title('True Solutions')
plt.show()


# In[] Import NN packages
import time
start = time.time()

# GPU usage
import keras
import tensorflow as tf
config = tf.ConfigProto( device_count = {'GPU': 1 , 'CPU': 8} ) 
sess = tf.Session(config=config) 
keras.backend.set_session(sess)



from keras.layers import Layer
import keras.backend as K

from keras.models import Sequential  #a linear stack of layers

#Before training a model, you need to configure the learning process
#compile method= optimizer +  loss function + list of metrics

from keras import optimizers
import numpy as np
import matplotlib.pyplot as plt

# In[] RNN initializaiton
import random

# self-defined guessing estmate parameter magnitude (intialization)



K0 =[0,0,0,0,0.21,0,0.31,0.11]
KA =[0.479077,0.212923,0.505692,0.0532308,0.0798462,0.1730000,-0.319385,-0.133077] 
C_kaiC = 3.4
C_kaiA = 1.3
m = 1
K_half = 0.43


a1=0.3
a2=3

K0_ran=[K0[i]*random.uniform(a1,a2) for i in range(len(K0))]
KA_ran=[KA[i]*random.uniform(a1,a2) for i in range(len(KA))]


C_kaiC = 3.4
C_kaiA = 1.3
m = 1
K_half = 0.43


p_ran=K0_ran+KA_ran

p=p_ran+[C_kaiC]+[C_kaiA]+[m]+[K_half]
np.save('initial_p.npy',p_ran)
# In[]
def my_init(shape, dtype=None): 
    return K.variable(p_ran)



# In[] Initial Guessing

# ODE solver parameters
abserr = 1.0e-8
relerr = 1.0e-6
stoptime = 100
numpoints = 1000

# Create the time samples for the output of the ODE solver.
t = [stoptime * float(i) / (numpoints - 1) for i in range(numpoints)]
#t=np.asarray(t)


#initial conditions
w0 = [0.68, 1.36,0.34]

# Call the ODE solver.
wsol_random = odeint(vectorfield, w0, t, args=(p,),
              atol=abserr, rtol=relerr)

np.save('wsol_random.npy',wsol_random)
#numerical dydt

del abserr,relerr,numpoints,stoptime  


plt.figure()

plt.plot(exp.keys(), [i[0] for i in exp.values()], 'o', color='blue')
plt.plot(exp.keys(), [i[1] for i in exp.values()], 'o', color='green')
plt.plot(exp.keys(), [i[2] for i in exp.values()], 'o', color='red')
plt.plot(t, wsol_random[:,0], color='blue')
plt.plot(t,wsol_random[:,1], color='green')
plt.plot(t,wsol_random[:,2], color='red')
plt.show()

plt.savefig('initial.png')


# In[] Construct the ODE layer
class ODE_RNN(Layer):
    
    
    
    def __init__(self, steps, h, **kwargs):
        self.steps = steps
        self.h = h
        super(ODE_RNN, self).__init__(**kwargs)

    def build(self, input_shape): # 
        self.kernel = self.add_weight(name='kernel', 
                                      shape=(20,),
                                      initializer=my_init,
                                      trainable=True)
        

#most critical function in the class        
    def step_do(self, step_in, states): # iterarion
        x = states[0]
    
        p = [ self.kernel[0], self.kernel[1],
              self.kernel[2], self.kernel[3],
              self.kernel[4], self.kernel[5],
              self.kernel[6], self.kernel[7],
              self.kernel[8], self.kernel[9],
              self.kernel[10], self.kernel[11],
              self.kernel[12], self.kernel[13],
              self.kernel[14], self.kernel[15],]
              
        
        p= tf.cast(p, tf.float32)
        T,D,S = x[:,0],x[:,1],x[:,2]  
        
        #p=K0+KA+[C_kaiC]+[C_kaiA]+[m]+[K_half]
# relate paramter space and functional relations   
        
        
        K0=p[0:8]          # parameter space
        KA=p[8:16]
        
        C_kaiC = 3.4
        C_kaiA = 1.3
        m = 1
        K_half = 0.43


        A = tf.maximum(0.0, C_kaiA-2*m*S)
        
        Kut=K0[0] + KA[0]*A/(K_half + A)
        Ktd=K0[1] + KA[1]*A/(K_half + A)
        Ksd=K0[2] + KA[2]*A/(K_half + A)
        Kus=K0[3] + KA[3]*A/(K_half + A)
        Ktu=K0[4] + KA[4]*A/(K_half + A)
        Kdt=K0[5] + KA[5]*A/(K_half + A)
        Kds=K0[6] + KA[6]*A/(K_half + A)
        Ksu=K0[7] + KA[7]*A/(K_half + A)
        
            
        #[Kut,Ktd,Ksd,Kus,Ktu,Kdt,Kds,Ksu] = K0 + np.multiply(KA,A)/(K_half + A)
        
        
        U = C_kaiC -(T+D+S)
        y0 = Kut * U + Kdt * D - Ktu * T - Ktd * T
        y1 = Ktd * T + Ksd * S - Kdt * D - Kds * D
        y2 = Kus * U + Kds * D - Ksu * S - Ksd * S
    
    
    

        
        

        yy=[y0,y1,y2]
          

        for a in range(3):
            yy[a]= eval('K.expand_dims(y{}, 1)'.format(a)) 



        y = K.concatenate([yy[0], yy[1]], 1)
        y = K.concatenate([y, yy[2]], 1)


        
        
        step_out = x + self.h * K.clip(y, -1e5, 1e5) 
        return step_out, [step_out]



    def call(self, inputs): # initial condition
        init_states = [inputs]
        zeros = K.zeros((K.shape(inputs)[0],
                         self.steps,
                         K.shape(inputs)[1])) # 
        
        outputs = K.rnn(self.step_do, zeros, init_states) # 
        return outputs[1] # 

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.steps, input_shape[1])


# In[] USE RNN to infer the parameters


#steps,h = 100, 1 # big step
steps,h = 100, 1 # big step

series=exp # suppose it's from exp dat
n=len(w0)     #number of species


M = Sequential()

M.add(ODE_RNN(steps, h, input_shape=(n,)))  #add self-defined layers


M.summary()


# In[]
# X: initial conditions
# Y: subsequent 
X = np.array([series[0]])
Y = np.zeros((1, steps, n))

for i,j in series.items():
    if i != 0:
        Y[0, int(i/h)-1] += series[i]


# In[]
# Only consider point with data
def ode_loss(y_true, y_pred):
    T = K.sum(K.abs(y_true), 2, keepdims=True)
    T = K.cast(K.greater(T, 1e-3), 'float32')
    return K.sum(T * K.square(y_true - y_pred), [1,2])


# In[] Fitting
     
opt=optimizers.Adam(lr=1e-5)


M.compile(optimizer=opt, loss=ode_loss )


 
train_history =M.fit(X, Y, epochs=100000) #
loss = train_history.history['loss']

end = time. time()
print('runing time is', end - start, 's')

np.save('loss.npy',loss)



# In[]
# result
result = M.predict(np.array([w0]))[0]
times = np.arange(1, steps+1) * h
np.save('result.npy',result)

# In[] USE RNN to infer the parameters

infer=np.asarray(M.get_weights())
np.save('infer_p.npy',infer)

K0 =[0,0,0,0,0.21,0,0.31,0.11]
KA =[0.479077,0.212923,0.505692,0.0532308,0.0798462,0.1730000,-0.319385,-0.133077] 


p=K0 + KA
for i in range(len(p)):
    print("inferred values", infer[0,i], "true values", p[i])

#print("true values", p)


#dev=np.amax(abs((infer-p)/p))
#print('max deviation', dev)



# In[] Initial Guessing

# ODE solver parameters
abserr = 1.0e-8
relerr = 1.0e-6
stoptime = 100
numpoints = 1000

# Create the time samples for the output of the ODE solver.
t = [stoptime * float(i) / (numpoints - 1) for i in range(numpoints)]
#t=np.asarray(t)


p=p_ran+[C_kaiC]+[C_kaiA]+[m]+[K_half]

# Call the ODE solver.
wsol_random = odeint(vectorfield, w0, t, args=(p,),
              atol=abserr, rtol=relerr)

#numerical dydt

del abserr,relerr,numpoints,stoptime  


plt.figure()

no=[0,1,2]
no=np.sort(no)



lw = 2
plt.plot(exp.keys(), [i[no[0]] for i in exp.values()], 'o', color='blue',linewidth=lw)
plt.plot(exp.keys(), [i[no[1]] for i in exp.values()], 'o', color='green',linewidth=lw)
plt.plot(exp.keys(), [i[no[2]] for i in exp.values()], 'o', color='red',linewidth=lw)


plt.plot(t, wsol_random[:,no[0]], color='blue',label = r'y{}'.format(no[0]))
plt.plot(t,wsol_random[:,no[1]], color='green',label = r'y{}'.format(no[1]))
plt.plot(t,wsol_random[:,no[2]], color='red',label = r'y{}'.format(no[2]))

plt.title('initial guess')
plt.show()
plt.legend()
plt.savefig('initial.png')
bottom, top = plt.ylim()  # return the current ylim


# In[] Visualization
plt.figure()



plt.plot(times, result[:,no[0]], color='blue')
plt.plot(times, result[:,no[1]], color='green')
plt.plot(times, result[:,no[2]], color='red')

plt.plot(series.keys(), [i[no[0]] for i in series.values()], 'o', color='blue',label = r'y{}'.format(no[0]))
plt.plot(series.keys(), [i[no[1]] for i in series.values()], 'o', color='green',label = r'y{}'.format(no[1]))
plt.plot(series.keys(), [i[no[2]] for i in series.values()], 'o', color='red',label = r'y{}'.format(no[2]))
plt.show()
plt.ylim((bottom, top))

plt.legend()
plt.title('RNN result')
plt.savefig('test.png')











