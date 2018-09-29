import os
import pickle

import numpy as np


def esm_record(id_list, pred_list, path):
    '''
    传入id列表与预测列结果,为之后的集成学习准备数据
    :param id_list:
    :param pred_list:
    :param path:
    :return:
    '''
    if os.path.exists(path): # 如果文件已存在，则删除之前的，实现文件的刷新
        os.remove(path)
    with open(path,mode="wb") as f:
        obj={}
        for i in range(len(id_list)):
            p=pred_list[i]
            obj[id_list[i]]=p
        pickle.dump(obj,f)

