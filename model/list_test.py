import numpy as np

id_list=[]
pred_list=[]
for num, i in enumerate(range(0, 300,10)):
    print(i)
    pred=list(np.random.rand(10,3))
    ids=list(np.random.rand(10,1))

    id_list.extend(ids)
    pred_list.extend(pred)
print(id_list)
print(pred_list)