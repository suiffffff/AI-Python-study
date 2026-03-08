import numpy as np

test01="./testdata/test1.csv"

t1=np.loadtxt(test01,delimiter=",",dtype=float)
t2=np.loadtxt(test01,delimiter=",",dtype=float,unpack=True)

print(t1)
print(t2)
sp=np.array([0,3])
print(t1[sp])

arr=np.arange(25).reshape((5,5))
print(arr)
arr=np.where(arr<10,0,10)
print(arr)
