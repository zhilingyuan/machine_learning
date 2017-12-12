import numpy as np
iris=[1,2,0]
binary_target=np.array([1.if x==0 else 0. for x in iris])
print(binary_target)
