from numba import cuda, jit, float32
import numpy as np
import matplotlib.pyplot as plt
from time import time

## vector addition
@cuda.jit
def vector_add(x, y, out):
    
    i = cuda.grid(1)

    if i < x.size:
        out[i] = x[i] + y[i]

## vector subtraction
@cuda.jit
def vector_sub(x, y, out):
    
    i = cuda.grid(1)

    if i < x.size:
        out[i] = x[i] - y[i]

## matrix vector multiplication
@cuda.jit
def matrix_vector_mul(A, x, out):

    i = cuda.grid(1)

    if i < A.shape[0]:
        tmp = 0
        for j in range(A.shape[1]):
            tmp += A[i,j] * x[j]
        out[i] = tmp

## optimize function
@cuda.jit
def optimize(grad, lr, x):
    
    i = cuda.grid(1)

    if i < x.size:
        x[i] -= grad[i] * lr * 2

@jit(nopython=True)
def inner_product(x, y):
    out = 0
    
    for i in range(x.size):
        out += x[i] * y[i]

    return out

## inner product function
@jit(nopython=True)
def inner_product_for_grad(x, y, b):
    out = 0.
    
    for i in range(x.size):
        out += x[i] * y[i]
    
    out -= b

    return out

TPB = 16

## calculate gradient function
@cuda.jit
def get_grad(A, x, b, out):
    sA = cuda.shared.array(shape=(TPB,TPB), dtype=float32)
    sB = cuda.shared.array(shape=(TPB), dtype=float32)

    i = cuda.grid(1)

    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y

    if i >= out.shape[0]:
        ## Quit if (x) is outside of valid out boundary
        return

    tmp = 0.
    for j in range(int(A.shape[0] / TPB)):
        ## Preload data into shared memory
        sA[tx, ty] = A.T[i, ty + j * TPB]
        sB[tx] = inner_product_for_grad(A[tx + j * TPB,:], x, b[tx + j * TPB])

        ## Wait until all threads finish proloading
        cuda.syncthreads()

        ## Computes partial product on the shared memory
        for k in range(TPB):
            tmp += sA[tx, k] * sB[tx]

        ## Wait until all threads finish computing
        cuda.syncthreads()

    out[i] = tmp