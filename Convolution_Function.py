from numba import cuda, jit

@jit(nopython=True)
def matmul(data, weight):
    out = 0

    for x in range(data.shape[0]):
        for y in range(data.shape[1]):
            out += data[x,y] * weight[x,y]
    return out

@cuda.jit
def convolution_multiply(data, weight, stride, output):
    num, row, column = cuda.grid(3)
    fh, fw = weight.shape

    if num < output.shape[0] and row < output.shape[1] and column < output.shape[2]:
        i, i_max = stride*row, stride*row + fh
        j, j_max = stride*column, stride*column + fw
        
        output[num,row,column] = matmul(data[num,i:i_max,j:j_max], weight)
