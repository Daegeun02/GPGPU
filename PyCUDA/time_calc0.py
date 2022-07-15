import numpy as np
import pycuda.autoinit
from pycuda import gpuarray
from time import time

host_data = np.float32(np.random.random(50000000))

t1 = time()
host_data_2x = host_data * np.float32(2)
t2 = time()

print(f"total time to compute on CPU: {t2 - t1}")
device_data = gpuarray.to_gpu(host_data)

t1 = time()
device_data_2x = device_data * np.float32(2)
t2 = time()

from_device = device_data_2x.get()
print(f"total time to compute on GPU: {t2 - t1}")

print(f"Is the host computation the same as the GPU computation?: {np.allclose(from_device, host_data_2x)}")