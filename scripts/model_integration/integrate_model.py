from scipy.integrate import odeint
import scripts.model_compilation.model_func as model_func
import numpy as np 
import matplotlib.pyplot as plt
import time




t = np.linspace(0, 1000, 10001)
# model_func.model_func(1,np.array([1,0,0,1,0,0,0]),np.array([0.9,0.6,1.1,0.2,0.8]))
s = np.array([0.2,0,0,0.2,0,0,0,0]).astype(float)
k = np.array([0.2,0.1,0.1,0.2,0.8]).astype(float)
residence_time = 120
dt = t[1]-t[0]

decay_constant = 2**(-dt/residence_time)
tic = time.time()
sol = odeint(model_func.model_func, s , t, args=(k,decay_constant))
toc = time.time()

print(f"integral took {toc-tic} seconds")

for compounds in sol.T:
    plt.plot(t,compounds)
plt.show()



