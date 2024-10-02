#!/usr/bin/env python
# coding: utf-8

# In[150]:


import autograd as ag
import autograd.numpy as np
from autograd.numpy import linalg as LA
# import numpy as nump
import scipy

import random
import math
import pandas as pd
from scipy.linalg import eigh as largest_eigh
from qutip import *
from scipy import stats
from scipy.interpolate import griddata
from mpl_toolkits.mplot3d import Axes3D
from itertools import chain, combinations, combinations_with_replacement
import itertools

import time

import qutip as qt

from scipy.stats import unitary_group

import itertools
from itertools import chain, combinations, combinations_with_replacement, product


# In[24]:


def phip(N,q): ##generate the φ+ state for N quqits
    d=q**N
    index=list(product(list(range(0,q)),repeat=N))
    phip=1/d**(1/2)*sum([tensor(basis([q for i in range(N)],list(j)),basis([q for  i in range(N)],list(j))) for j in index])
    
    return phip
    
def op_ent(q,X,alpha='2'): ## compute operator entanglement of observable X across two quqits
    
    N=2
    d=q**N
    
    index=list(product(list(range(0,q)),repeat=N))
    phip=1/d**(1/2)*sum([tensor(basis([q for i in range(N)],list(j)),basis([q for  i in range(N)],list(j))) for j in index])
    choi=tensor(X,qeye([q for i in range(N)]))*phip
    reduced_choi=choi.ptrace([i for i in range(N//2)]+[i+N for i in range(N//2)]) ##Reduced choi Tr_(AA')
    
    if alpha=='2':
        s=entropy_linear(reduced_choi)
        
    if alpha=='1':
        s=entropy_vn(reduced_choi,sparse=True)
    
    return s

def op_ent_swap(q,X,alpha='2'): ## compute operator entanglement of observable X*swap across two quqits
    
    d=q**2
    N=2
    index=list(product(list(range(0,q)),repeat=N))
    
    phip=1/d**(1/2)*sum([tensor(basis([q for i in range(N)],list(j)),basis([q for  i in range(N)],list(j))) for j in index])
    choi=tensor(X,qeye([q for i in range(N)]))*phip
    reduced_choi=choi.ptrace([i for i in range(N//2)]+[i+N+N//2 for i in range(N//2)]) ##Reduced choi Tr_(AB')
    
    if alpha=='2':
        s=entropy_linear(reduced_choi)
        
    if alpha=='1':
        s=entropy_vn(reduced_choi,sparse=True)
    
    return s

def opentp(q,X): ## compute operator entangling power of X acrotss two quqits

    N=2
    d=q**N
    Es=1-1/d
    eop=op_ent(q,X)
    eops=op_ent_swap(q,X)
    Ep=1-(1-eop/Es)**2-(1-eops/Es)**2-2/d*eop/Es*eops/Es

    return Ep

def generate_matrices(dimension): #generates all matrices with one 1 and all other 0 for a given dimension
    matrices = []
    for i in range(dimension):
        for j in range(dimension):
            matrix = [[0] * dimension for _ in range(dimension)]
            matrix[i][j] = 1
            matrices.append(np.array(matrix))
    return matrices

def swap_gen(N,i,j,q): ### generates the swap operator S_{ij} where i and j are lists indicating the quqits to be swapped out of N total quqits
    
    if len(i)!=len(j):
        print("Number of qubits mismatch")
        return
    if len(i)+len(j)>N:
        print("Not enough qubits")
        return
    
    identity_list=[qt.qeye(q) for i in range(N)] ###list with 1 everywhere
    matrices_list=generate_matrices(q) ###list of all 2x2 matrices with one 1
    
    index=list(product(list(range(0,q**2)),repeat=len(i)))
    
    id_copy=identity_list.copy()
    swap_list=[]
    
    for q in index: 
        counter=0
        for m in range(len(i)):
            braket=qt.Qobj(matrices_list[q[counter]]) #some single qubit braket |a><b|
            id_copy[i[m]]=braket
            id_copy[j[m]]=braket.dag()
            counter+=1
        swap_list.append(tensor(id_copy))
    swap=sum(swap_list)
    
    return swap


# In[39]:


#Gradient descent-specific functions

def E_p_grad_st(U,q): ###Calculates d(lta)/dU*(U)
    
    #basic initialization of things we'll need
    N=2
    d = q**N
    swap_a_a_prime = swap_gen(2*N,[0],[2],q)
    identity = qeye([q for i in range(N)])
    U_dag = U.dag()
    
    #creating tensorized operators
    u_tens_u = tensor(U, U)
    id_tens_u_dag = tensor(identity, U_dag)
    
    #matrix product
    product = id_tens_u_dag*swap_a_a_prime*u_tens_u*swap_a_a_prime
    
    #taking partial trace
    traced_product = product.ptrace([0,1])
        
    gradst = (-2/(d**2))*traced_product #this is the Euclidean gradient

    return gradst


# In[156]:


def grad_asc(U, q, step_cutoff, conv_cutoff, grad_cutoff, time_cutoff):

    N=2
    step_size = 1 #unitary step size that goes inside exponential

    break_flag=False

    E_p_init = opentp(q,U)
    time_flag=[]
    # E_p_save=[]
    # u_save=U

    t1=time.time()


    while break_flag==False:
        # E_p_save.append(E_p_init)
        gamma = E_p_grad_st(U,q) #if this is the correct implementation of gamma, then
        G = gamma*U.dag() - U*gamma.dag() #G should be strictly imaginary
        P = qt.Qobj((scipy.linalg.expm(1*step_size*G.full())),dims=G.dims)
        Q = P*P
        U_up=Q*U
        U_down=P*U


        while E_p_init - opentp(q,U_up) >= step_size*0.5*np.real((G*G.dag()).tr()):
            P = Q
            Q = P*P
            step_size = 2*step_size
            U_up = Q*U

        while E_p_init - opentp(q,U_down) < 0.5*step_size*0.5*np.real((G*G.dag()).tr()):
            P = qt.Qobj(scipy.linalg.expm(-1*step_size*G.full()),dims=G.dims)
            step_size = 0.5*step_size
            U_down = P*U
            if step_size<=step_cutoff:
                break

        U = P*U

        E_p_fin = opentp(q,U)
        change=abs(E_p_init-E_p_fin)
        grad_hs=0.5*np.real((G*G.dag()).tr())

        if change<=conv_cutoff and grad_hs<=grad_cutoff:
            break_flag=True
        else:
            E_p_init=E_p_fin

        if time.time()-t1>time_cutoff:
            break_flag=True
            print("Time limit exceeded")
            break

    return E_p_fin, U, grad_hs


# In[197]:


### Gradient ascent for 2 quqits starting from random unitary

E_p_list=[]
riem_grad_norm_list=[]
perfect_list=[]
bound_list=[]
qmax=10
for q in range(2,qmax):
    N=2
    d=q**N
    U=qt.rand_unitary(d,dims=[[q for i in range(N)],[q for i in range(N)]])
    step_cutoff=10**(-8)
    conv_cutoff=10**(-8)
    grad_cutoff=10**(-2)
    time_cutoff=600000

    E_p_fin, U_fin, riem_grad = grad_asc(U, q, step_cutoff, conv_cutoff, grad_cutoff,time_cutoff)

    E_p_list.append(E_p_fin)
    riem_grad_norm_list.append(riem_grad)

    if q == 2:
        perfect_list.append('None')
    else:
        perfect_list.append(1-2/(d))
    bound_list.append(1-2/(d+1))


# In[200]:


##export data

quqitsizes=[i for i in range(2,qmax)]
dgrad={'Quqit size q':quqitsizes}
dgrad.update({'Max Operator entangling power':E_p_list, 'Riemannian gradient HS norm squared at max':riem_grad_norm_list, 'Perfect Tensor Operator entangling power':perfect_list,'Upper Bound Operator entangling power':bound_list})
dgrad=pd.DataFrame(dgrad)
dgrad.to_csv("gradient_ascent.csv",index=False)


# In[ ]:




