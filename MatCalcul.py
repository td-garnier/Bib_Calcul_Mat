import numpy as np
from random import randint
from numba import njit, jit, int32, float32, vectorize,prange
import time
import pandas as pd

# Time Mesurement for Matrix
class Time_Measurement:
    def __init__(self):
        pass

    def matrix_operations_times(self,matrix,size,optimizer=None):
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

    def compare_for_severals_matrix_sizes(self,matrix,square_size,optimizer=None):
        result={}
        self.matrix_operations_times(matrix,size=(1,1),optimizer=optimizer)
        for i in square_size:
            result[i]=self.matrix_operations_times(matrix,size=(i,i),optimizer=optimizer)
        return pd.DataFrame(result)

## No Numpy
class Matrix:
    def __init__(self,array=None,size=(3,3),optimizer=None):
        
        self.array_generation(array,size,optimizer)
    
        match optimizer:
            case 'numpy-jit':
                self.add=np_add_jit
                self.mul=np_mul_jit
            case 'numpy-njit':
                self.add=np_add_njit
                self.mul=np_mul_njit
            case 'njit-parallel':
                self.add=add_njit
                self.mul=mul_njit
            case _:
                self.add=add
                self.mul=mul
        
    def __str__(self):
        return '{}'.format(self.values)

    def __add__(self,B):
        return self.add(self.values,B.values)
    
    def __mul__(self,B):
        return self.mul(self.values,B.values)
    
    def array_generation(self,array=None,size=(3,3),optimizer=None):
        if optimizer == None or optimizer =='njit-parallel':
            if array==None:
                self.values=[[0 for _ in range(size[0])] for _ in range(size[1])]
            else:
                self.values=array
        elif optimizer == 'numpy-jit' or optimizer == 'numpy-njit':
            if array==None:
                self.values=np.zeros(size,dtype=np.float32)
            else:
                self.values=np.array(array,dtype=np.float32)

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








