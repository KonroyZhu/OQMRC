import json
import numpy as np


import tensorflow as tf
import pickle

from train import train, test
from utils.ensemble_record import esm_record
from utils.utils import shuffle_data, padding, pad_answer

data_path="data/"
# vocab_size=process_data(data_path=data_path,threshold=5)
vocab_size = 98745

with open(data_path + 'train.pickle', 'rb') as f:
    train_data = pickle.load(f)
with open(data_path + 'dev.pickle', 'rb') as f:
    dev_data = pickle.load(f)
with open(data_path + 'testa.pickle', 'rb') as f:
    test_data = pickle.load(f)
dev_data = sorted(dev_data, key=lambda x: len(x[1]))
opts=json.load(open("model/config.json"))
print('train data size {:d}, dev data size {:d}, testa data size {:d}'.format(len(train_data), len(dev_data),len(test_data)))

def load_model(session,path="net/rnet/"):
    if not path.endswith("/"): path+="/"
    print("loading graph...")
    saver=tf.train.import_meta_graph(path+"model.ckpt.meta")
    print("restore...")
    saver.restore(session, path+"model.ckpt")  # 到.ckpt即可，saver中的命名一致
    graph = tf.get_default_graph()
    # placeholder
    print("getting placeholder...")
    para=graph.get_tensor_by_name("para:0")
    que=graph.get_tensor_by_name("query:0")
    ans=graph.get_tensor_by_name("ans:0")
    # pred
    print("getting pred...")
    pred=graph.get_tensor_by_name("pred:0")
    # loss
    print("getting loss...")
    loss=graph.get_tensor_by_name("Mean:0")
    # optimizer
    print("getting optimizer...")
    optimizer=graph.get_operation_by_name("opt")

    return optimizer,loss,pred,para,que,ans


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    opt,los,pr,pa,qu,an=load_model(sess,path="net/rnet")
    tensor_dict={
        "p":pa,
        "q":qu,
        "a":an
    }

    best = test(pr,sess,tensor_dict)
    print("best from last epoch: {}".format(best))
    epoch = 0
    acc = 0
    # 储存
    saver = tf.train.Saver()

    for e in range(2):
        train_id, train_pred=train(epoch=e,session=sess,loss=los,optimizer=opts,tensor_dict=tensor_dict)
        acc, dev_id, dev_pred = test(pred=pr, session=sess, tensor_dict=tensor_dict)
        print("epoch {} acc: {} best {}".format(e, acc, best))
        if acc > best:
            best = acc
            print("saving...")
            # with open(args.save, 'wb') as f: TODO: 如何保存tf模型
            #     torch.save(model, f)save
            saver.save(sess, save_path="net/model.ckpt")  # save方法自带刷新功能
            esm_record(id_list=train_id, pred_list=train_pred, path="esm_record/train.pkl")
            esm_record(id_list=dev_id, pred_list=dev_pred, path="esm_record/dev.pkl")

    print('epcoh {:d} dev acc is {:f}, best dev acc {:f}'.format(e, acc, best))


