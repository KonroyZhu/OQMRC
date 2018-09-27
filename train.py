import json
# import  numpy as np
import numpy as np

from model.MwAN_full import MwAN
# from prepro.preprocess import process_data
import tensorflow as tf
import pickle

from model.R_Net import R_Net
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


def test_trainer(epoch,session,test_op ,tensor_dict):
    """
    快速检测输入数据是否有误，避免 setting an array element with a sequence.
    """
    data = train_data
    exception = []
    for num, i in enumerate(range(0, len(data), opts["batch"])):
        one = data[i:i + opts["batch"]]
        query, _ = padding([x[0] for x in one], max_len=opts["q_len"])
        passage, _ = padding([x[1] for x in one], max_len=opts["p_len"])
        answer = pad_answer([x[2] for x in one],max_len=opts["alt_len"])
        # query, passage, answer = np.array(query), np.array(passage), np.array(answer)
        fd = {
            tensor_dict["p"]: passage,
            tensor_dict["q"]: query,
            tensor_dict["a"]: answer
        }
        print("data idx: {} ".format(i))
        l = 0
        try:
            t_op = session.run(test_op, feed_dict=fd)
        except Exception as e:
            print('Error:', e)
            ex=""
            try:
                sess.run(test_op[0],feed_dict={tensor_dict['p']:passage})
            except Exception as e:
                ex="passage"
                print('Error in passage:', e)
            try:
                sess.run(test_op[1],feed_dict={tensor_dict['a']:answer})
            except Exception as e:
                ex="answer"
                print('Error in answer:', e)
            try:
                sess.run(test_op[2],feed_dict={tensor_dict['q']:answer})
            except Exception as e:
                ex="query"
                print('Error in query:', e)
            exception.append(str(i)+ex)

    print("exception ids: {}".format(" ".join([str(id)for id in exception])))

def train(epoch,session,loss,optimizer,tensor_dict):
    data = shuffle_data(train_data, 1)
    total_loss = 0.0
    exception=[]
    for num, i in enumerate(range(0, len(data), opts["batch"])):
        one = data[i:i + opts["batch"]]
        query, _ = padding([x[0] for x in one], max_len=opts["q_len"])
        passage, _ = padding([x[1] for x in one], max_len=opts["p_len"])
        answer = pad_answer([x[2] for x in one],max_len=opts["alt_len"])
        # query, passage, answer = np.array(query), np.array(passage), np.array(answer)
        fd={
            tensor_dict["p"]:passage,
            tensor_dict["q"]:query,
            tensor_dict["a"]:answer
        }
        print("data idx: {} ".format(i))
        l=0
        try:
            _,l = session.run([optimizer,loss],feed_dict=fd)
        except Exception as e:
            print('Error:', e)
            print("id: {} query: {}, passage {}, answer {}".format(i, np.shape(query), np.shape(passage),
                                                                   np.shape(answer)))
            exception.append(i)
        total_loss += l
        if (num + 1) % opts["interval"]== 0:
            print('|------epoch {:d} train error(total ) is {:f}  progress {:.2f}%------|'.format(epoch,
                                                                                   total_loss / opts["interval"],
                                                                                   i * 100.0 / len(data)))
    print("exception ids: {}".format(" ".join(exception)))
    total_loss = 0

def test(pred,session,tensor_dict):
    r, a = 0.0, 0.0
    for i in range(0, len(dev_data),opts["batch"]):
        one = dev_data[i:i +opts["batch"]]
        query, _ = padding([x[0] for x in one], max_len=opts["q_len"])
        passage, _ = padding([x[1] for x in one], max_len=opts["p_len"])
        answer = pad_answer([x[2] for x in one],max_len=opts["alt_len"])
        # query, passage, answer = np.array(query), np.array(passage), np.array(answer)
        fd = {
            tensor_dict["p"]: passage,
            tensor_dict["q"]: query,
            tensor_dict["a"]: answer
        }
        p = session.run([pred], feed_dict=fd)
        r=0
        for item in p:
            if item == 0:
                r+=1
        a += len(one)
    return r * 100.0 / a

if __name__ == '__main__':
    model=MwAN()
    # model = R_Net()
    loss, optimizer, predict, tensor_dict, test_op = model.build()

    best = 0.0
    sess=tf.Session()
    sess.run(tf.global_variables_initializer())
    epoch=0
    acc=0
    # 储存
    saver=tf.train.Saver()
    # """ 用于切换测试与运行模式
    print("正在测试feed_dict输入数据...")
    test_trainer(epoch, sess, test_op, tensor_dict)
    """
    for epoch in range(opts["epoch"]):
        train(epoch,sess,loss,optimizer,tensor_dict)
        acc = test(predict,sess,tensor_dict)
        if acc > best:
            best = acc
            # with open(args.save, 'wb') as f: TODO: 如何保存tf模型
            #     torch.save(model, f)save
            saver.save(sess,save_path="net/model.ckpt") # save方法自带刷新功能

    print('epcoh {:d} dev acc is {:f}, best dev acc {:f}'.format(epoch, acc, best))
    # """