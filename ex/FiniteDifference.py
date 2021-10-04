#!/usr/bin/env python
# coding: utf-8

# In[5]:


import numpy as np
from math import sqrt
import scipy.integrate
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML
from scipy import sparse
from scipy.sparse.linalg import eigs, spsolve
from tqdm.notebook import trange, tqdm
from nodepy.runge_kutta_method import *

"""
Fucntions that are useful to genrate the finite difference matrices
"""


def laplacian_1D(m):
    em = np.ones(m)
    e1=np.ones(m-1)
    A = (sparse.diags(-2*em,0)+sparse.diags(e1,-1)+sparse.diags(e1,1))/((1/(m+1))**2);
    return A;
def laplacian_1D_with_reaction(m):
    em = np.ones(m)
    e1=np.ones(m-1)
    A = (sparse.diags(-2*em,0)+sparse.diags(e1,-1)+sparse.diags(e1,1))/((1/(m+1))**2);
    A = A -sparse.diags(1*em,0)
    return A;
def laplacian_2D(m):
    I = np.eye(m)
    A = laplacian_1D(m)
    return sparse.kron(A,I) + sparse.kron(I,A)

"""
Solver they get in input the number of point in the mesh and the forcing function f.
"""
def Solve1DLaplace(m,f):
    x=np.linspace(0,1,m+2); 
    y=x[1:-1]
    F=f(y);
    K = -laplacian_1D(m);
    u = spsolve(K,F);
    u =np.hstack(([0.0],u,[0.0]))
    print(len(x),len(u))
    return u, x;
def Solve1DLaplaceReaction(m,f):
    x=np.linspace(0,1,m+2); 
    y=x[1:-1]
    F=f(y);
    K = -laplacian_1D_with_reaction(m);
    u = spsolve(K,F);
    u =np.hstack(([0.0],u,[0.0]))
    print(len(x),len(u))
    return u, x;
def Solve2DLaplace(m,f):
    x=np.linspace(0,1,m+2); x=x[1:-1]
    y=np.linspace(0,1,m+2); y=y[1:-1]
    X,Y=np.meshgrid(x,y)
    F = f(X,Y);
    F = F.reshape(m**2);
    K=laplacian_2D(m)
    u =-spsolve(K,F);
    u = u.reshape([m,m])
    return u, (X,Y);


# ### Simple Laplace Problem 1D

# In[2]:


def f(x):
    return -6*x+2;
def exact(x):
    return (x**2)*(x-1);
u,x = Solve1DLaplace(100,f)
plt.plot(x,u);
plt.plot(x,exact(x),"--")
plt.legend(["Numerical Solution","Analytical"])


# ### Simple Elliptic Reaction Diffusion Problem 1D

# In[3]:


def f(x):
    return (x**2)*(x-1)-6*x+2;
def exact(x):
    return (x**2)*(x-1);
u,x = Solve1DLaplaceReaction(100,f)
plt.plot(x,u);
plt.plot(x,exact(x),"--")
plt.legend(["Numerical Solution","Analytical"])


# ### Simple Laplace Problem 2D

# In[6]:


def f(x,y):
    return -(2*(y**2)-2*y+2*(x**2)-2*x);
def exact(x,y):
    return x*(x-1)*y*(y-1);
u,grid = Solve2DLaplace(100,f)
plt.pcolor(grid[0],grid[1],u)
plt.colorbar()
plt.figure()
plt.pcolor(grid[0],grid[1],exact(grid[0],grid[1]));
plt.colorbar()

