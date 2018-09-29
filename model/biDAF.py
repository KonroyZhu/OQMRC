import json
import  tensorflow as tf

class BiDAF():
    def __init__(self):
        self.opts = json.load(open("model/config.json"))
        opts = self.opts

    def build(self):
        print("building model...")
        opts = self.opts

        # placeholder
        # FIXME: batch should be changed to None
        query = tf.placeholder(dtype=tf.int32, shape=[opts["batch"], opts["q_len"]], name="query")
        para = tf.placeholder(dtype=tf.int32, shape=[opts["batch"], opts["p_len"]], name="para")
        ans = tf.placeholder(dtype=tf.int32, shape=[opts["batch"], 3, opts["alt_len"]],
                             name="ans")  # 每个ans中有三个小句，第一句为正确答案 FIXME: alt_len should be None
