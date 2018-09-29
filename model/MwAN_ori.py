import math

import tensorflow as tf
import json

class MwAN_ori:
    def __init__(self):
        self.opts = json.load(open("model/config.json"))
        opts = self.opts


        # self.embedding = nn.Embedding(vocab_size + 1, embedding_dim=embedding_size)
        self.emb = tf.Variable(tf.truncated_normal(shape=[98745 + 1, opts["embedding_size"]], mean=0.0, stddev=1.0,dtype=tf.float32))

        # self.a_attention = nn.Linear(embedding_size, 1, bias=False)
        stdv = 1. / math.sqrt(opts["embedding_size"])
        self.a_attention = tf.Variable(tf.random_uniform(minval=-stdv,maxval=stdv,shape=[opts["embedding_size"],1]))
    def weight_matmul(self,mat,weight):


    def build(self):
        print("building model...")
        opts = self.opts

        # placeholder
        # FIXME: batch should be changed to None
        query = tf.placeholder(dtype=tf.int32, shape=[opts["batch"], opts["q_len"]], name="query")
        passage = tf.placeholder(dtype=tf.int32, shape=[opts["batch"], opts["p_len"]], name="para")
        answer = tf.placeholder(dtype=tf.int32, shape=[opts["batch"], 3, opts["alt_len"]],name="ans")

        q_embedding = tf.nn.embedding_lookup(params=self.emb,ids=query)
        p_embedding = tf.nn.embedding_lookup(params=self.emb,ids=passage)
        a_embeddings = tf.nn.embedding_lookup(params=self.emb,ids=answer)

        print("layer1: encoding layer")
        with tf.variable_scope("qp_encoding"):
            cell_fw=tf.nn.rnn_cell.GRUCell(num_units=opts["hidden_size"])
            cell_bw=tf.nn.rnn_cell.GRUCell(num_units=opts["hidden_size"])
            q_encoder,_=tf.nn.bidirectional_dynamic_rnn(cell_fw,cell_bw,q_embedding,dtype=tf.float32) # (b,q,2h)
            q_encoder=tf.concat(q_encoder,axis=2)
            p_encoder,_=tf.nn.bidirectional_dynamic_rnn(cell_fw,cell_bw,p_embedding,dtype=tf.float32) # (b,p,2h)
            p_encoder=tf.concat(p_encoder,axis=2)
            print("p/q_encoder: {}".format(q_encoder))

        with tf.variable_scope("a_encoding"):
            cell_fw_a = tf.nn.rnn_cell.GRUCell(num_units=opts["hidden_size"]/2)
            cell_bw_a = tf.nn.rnn_cell.GRUCell(num_units=opts["hidden_size"]/2)
            a_embeddings_reshaped=tf.reshape(a_embeddings,shape=[-1,3,opts["embedding_size"]]) # (b*a,3,d)
            a_encoder,_=tf.nn.bidirectional_dynamic_rnn(cell_fw_a,cell_bw_a,a_embeddings_reshaped,dtype=tf.float32)
            a_encoder=tf.concat(a_encoder,axis=2) # (b,a,h)
            print("a_encoder: {}".format(a_encoder))
            a_score = tf.nn.softmax(, 1)
            a_output = a_score.transpose(2, 1).bmm(a_embedding).squeeze()
            a_embedding = a_output.view(a_embeddings.size(0), 3, -1)


