#%%
import numpy as np
import pickle
import gzip
import cca_core
from CKA import linear_CKA, kernel_CKA

#%%
# verify that the CKA is working
X = np.random.randn(100, 64)
Y = np.random.randn(100, 64)

print('Linear CKA, between X and Y: {}'.format(linear_CKA(X, Y)))
print('Linear CKA, between X and X: {}'.format(linear_CKA(X, X)))

print('RBF Kernel CKA, between X and Y: {}'.format(kernel_CKA(X, Y)))
print('RBF Kernel CKA, between X and X: {}'.format(kernel_CKA(X, X)))
# %%
# The minist layers are: 784(input)--500--500--10(output)
# check if CKA works for mnist data
# Load up second hidden layer of MNIST networks and compare
with open("model_activations/MNIST/model_0_lay01.p", "rb") as f:
    acts1 = pickle.load(f)
with open("model_activations/MNIST/model_1_lay01.p", "rb") as f:
    acts2 = pickle.load(f)
    
print("activation shapes", acts1.shape, acts2.shape)

#results = cca_core.get_cca_similarity(acts1, acts2, epsilon=1e-10, verbose=False)
    
# %%
# The problem of CKA: time-consuming with large data points
print('Linear CKA: {}'.format(linear_CKA(acts1.T, acts2.T)))
print('RBF Kernel: {}'.format(kernel_CKA(acts1.T, acts2.T)))
# %%
# similarity index by CCA
results = cca_core.get_cca_similarity(acts1, acts2, epsilon=1e-10, verbose=False)
print("Mean CCA similarity", np.mean(results["cca_coef1"]))
# %%
