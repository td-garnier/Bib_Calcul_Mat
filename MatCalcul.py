import numpy as np
from random import randint
from numba import njit, jit, int32, float32, vectorize,prange,cuda
import time
import pandas as pd
import math
import cupy as cp
import matplotlib.pyplot as plt

# Time Mesurement for Matrix
class Time_Measurement:
    def __init__(self):
        pass

    def _matrix_operations_times(self,matrix,size,optimizer=None):
        A = matrix(size=size,optimizer=optimizer)
        B = matrix(size=size,optimizer=optimizer)
        result={}
        start = time.time()
        A+B
        result['add']=time.time()-start
        start = time.time()
        A*B
        result['mul']=time.time()-start
        return result

    def _compare_for_severals_matrix_sizes(self,matrix,square_size,optimizer=None):
        result={}
        self._matrix_operations_times(matrix,size=(1,1),optimizer=optimizer)
        for i in square_size:
            result[i]=self._matrix_operations_times(matrix,size=(i,i),optimizer=optimizer)
        return result
    
    def get_times_for_severals_matrix_sizes(self,matrix,square_size,optimizer=None):
        return pd.DataFrame(self._compare_for_severals_matrix_sizes(matrix,square_size,optimizer))
    
    def _compare_for_severals_matrix_sizes_and_optimizer(self,matrix,square_size,optimizer_list):
        result={}
        for optimizer in optimizer_list:
            result[optimizer]=self._compare_for_severals_matrix_sizes(matrix,square_size,optimizer)
        return result
    

    def plot_times_for_severals_matrix_sizes_and_optimizer(self,result_matrix):

        sizes = list(result_matrix[list(result_matrix.keys())[0]].keys())
        add_result={}
        mul_result={}
        # Résultats pour l'addition
        for key in result_matrix:
            add_result[key] = [result_matrix[key][size]['add'] for size in sizes]
            mul_result[key] = [result_matrix[key][size]['mul'] for size in sizes]

        fig,ax=plt.subplots(1,2,figsize=(12,4))

        ax[0].plot(pd.DataFrame(add_result,index=sizes))
        ax[0].set_xscale('log')
        ax[0].set_yscale('log')
        ax[0].set_xlabel('matrix size')
        ax[0].set_ylabel('time')
        ax[0].legend(add_result.keys())
        ax[0].set_title('add')

        ax[1].plot(pd.DataFrame(mul_result,index=sizes))

        ax[1].set_xscale('log')
        ax[1].set_yscale('log')
        ax[1].set_xlabel('matrix size')
        ax[1].set_ylabel('time')
        ax[1].legend(mul_result.keys())
        ax[1].set_title('mul')

        plt.tight_layout()
        plt.show()

## No Numpy
class Matrix:
    def __init__(self,array=None,size=(3,3),optimizer=None):
        self.optimizer=optimizer
        
        self.array_generation(array,size)
    
        match self.optimizer:
            case 'numpy-jit':
                self.add=np_add_jit
                self.mul=np_mul_jit
            case 'numpy-njit':
                self.add=np_add_njit
                self.mul=np_mul_njit
            case 'njit-parallel':
                self.add=add_njit
                self.mul=mul_njit
            case 'cuda' :
                self.add=add_cuda
                self.mul=mul_cuda
            case 'cupy' :
                self.add=add_cupy
                self.mul=mul_cupy
            case _:
                self.add=add
                self.mul=mul
        
    def __str__(self):
        return '{}'.format(self.values)

    def __add__(self,B):
        if self.check_same_optimizer(B):
            return self.add(self.values,B.values)
    
    def __mul__(self,B):
        if self.check_same_optimizer(B):
            return self.mul(self.values,B.values)
    
    def array_generation(self,array=None,size=(3,3)):
        if self.optimizer == None or self.optimizer =='njit-parallel':
            if array==None:
                self.values=[[0 for _ in range(size[0])] for _ in range(size[1])]
            else:
                self.values=array
        elif self.optimizer == 'numpy-jit' or self.optimizer == 'numpy-njit' or self.optimizer == 'cuda' :
            if array==None:
                self.values=np.zeros(size,dtype=np.float32)
            else:
                self.values=np.array(array,dtype=np.float32)
        elif self.optimizer == 'cupy':
            if array==None:
                self.values=cp.zeros(size,dtype=np.float32)
            else:
                self.values=cp.array(array,dtype=np.float32)
        else : 
            if array==None:
                self.values=[[0 for _ in range(size[0])] for _ in range(size[1])]
            else:
                self.values=array
    
    def check_same_optimizer(self,B):
        if self.optimizer==B.optimizer:
            return True
        else:
            print('Matrix must be generated with same optimizer')
            return False

@jit
def np_add_jit(A,B):
    return np.add(A,B)

@jit
def np_mul_jit(A,B):
    return np.dot(A,B)

@njit
def np_add_njit(A,B):
    return np.add(A,B)

@njit
def np_mul_njit(A,B):
    return np.dot(A,B)


def add(A,B):
    if len(A[0])==len(B[0]):
        result_l1=[]
        for i,ii in zip(range(len(A)),range(len(B))):
            result_l2=[]
            for val1,val2 in zip(A[i],B[ii]):
                result_l2.append(val1+val2)
            result_l1.append(result_l2)
        return result_l1
    else:
        print('Dimensions des matrices différentes')
    
def mul(A,B):
    B=list(zip(*B))
    if len(A[0])==len(B[0]):
        A_size=range(len(A))
        B_size=range(len(B))
        C=list(zip(A,list(B)))
        result_l1=[]
        for i in A_size:
            result_l2=[]
            for ii in B_size:
                result_l3=0
                for val1,val2 in zip(C[i][0],C[ii][1]):
                    result_l3+=val1*val2
                result_l2.append(result_l3)
            result_l1.append(result_l2)
        return result_l1

    else:
        print('Dimensions des matrices différentes')

def add_njit(A,B):
    if len(A[0])==len(B[0]):
        result_l1=[]
        for i,ii in zip(prange(len(A)),prange(len(B))):
            result_l2=[]
            for val1,val2 in zip(A[i],B[ii]):
                result_l2.append(val1+val2)
            result_l1.append(result_l2)
        return result_l1
    else:
        print('Dimensions des matrices différentes')
    
def mul_njit(A,B):
    mat1=np.array(A,dtype=np.float32)
    mat2=np.array(B,dtype=np.float32)
    nrows_mat1 = len(mat1)
    nrows_mat2 = len(mat2)
    ncols_mat1 = len(mat1[0])
    ncols_mat2 = len(mat2[0])
    if nrows_mat1 != ncols_mat2 or nrows_mat2 != ncols_mat1:
        raise ValueError("Cannot multiply")
    result = np.zeros((nrows_mat1,ncols_mat2),dtype=np.float32)
    result = mul_inside_jit(result, mat1, mat2, nrows_mat1, ncols_mat2, ncols_mat1)
    return result

@njit(parallel=True)
def mul_inside_jit(result, mat1, mat2, nrows_mat1, ncols_mat2, ncols_mat1):
    for i in prange(nrows_mat1):
        for j in prange(ncols_mat2):
            for k in prange(ncols_mat1):
                result[i][j] += mat1[i][k] * mat2[k][j]
    return result

def cuda_init(mat1,mat2):
    z_h = np.zeros(shape=(mat1.shape[0],mat2.shape[1]))   
    x_d = cuda.to_device(mat1)
    y_d = cuda.to_device(mat2)
    z_d = cuda.to_device(z_h)    
    threadsperblock = (16, 16)
    blockspergrid_x = math.ceil(z_h.shape[0] / threadsperblock[0])
    blockspergrid_y = math.ceil(z_h.shape[1] / threadsperblock[1])
    blockspergrid = (blockspergrid_x, blockspergrid_y)
    return threadsperblock,blockspergrid,x_d,y_d,z_d

# Test addition
def add_cuda(mat1,mat2):
    threadsperblock,blockspergrid,x_d,y_d,z_d=cuda_init(mat1,mat2)
    addition_cuda[blockspergrid, threadsperblock](x_d, y_d, z_d)    
    z_h = z_d.copy_to_host()
    return z_h

# CUDA kernel for matrix addition
@cuda.jit
def addition_cuda(mat1, mat2,new_matrix):
    x, y = cuda.grid(2)
    if x < new_matrix.shape[0] and y < new_matrix.shape[1]:
        new_matrix[x,y] = mat1[x, y] + mat2[x, y]

def mul_cuda(mat1,mat2):
    threadsperblock,blockspergrid,x_d,y_d,z_d=cuda_init(mat1,mat2)   
    multiplication_cuda[blockspergrid, threadsperblock](x_d, y_d, z_d)    
    z_h = z_d.copy_to_host()
    return z_h

# CUDA kernel for matrix multiplication
@cuda.jit
def multiplication_cuda(mat1,mat2, new_matrix):
    x, y = cuda.grid(2)
    if x < new_matrix.shape[0] and y < new_matrix.shape[1]:
        tmp = 0
        for k in range(mat1.shape[1]):
            tmp += mat1[x, k] * mat2[k, y]
        new_matrix[x,y] = tmp


# Fonction pour l'addition
def add_cupy(mat1,mat2):
    result=cp.add(mat1, mat2)
    cp.cuda.Stream.null.synchronize()
    return result


# Fonction pour la multiplication
def mul_cupy(mat1, mat2):
    result=cp.matmul(mat1, mat2)
    cp.cuda.Stream.null.synchronize()
    return result







