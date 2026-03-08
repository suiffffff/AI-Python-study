import numpy
import numpy as np

t1=np.array([1,2,3])
print(t1)
print(type(t1))

t2=numpy.arange(10)
print(t2)
print(t2.dtype)

t3=np.array(range(1,5),dtype="i1")
print(t3.dtype)

t4=np.arange(24).reshape((2,3,4))
print(t4)
print(t4.reshape(4,6))