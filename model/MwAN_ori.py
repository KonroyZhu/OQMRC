import math

import tensorflow as tf
import json

class MwAN_ori:

    def weight_matmul(self, mat, weight):
        """
        :param mat: (b,t,h)
        :param weight: (h,o)
        :return: (b,t,o)
        """
        m_s = mat.get_shape().as_list()
        w_s = weight.get_shape().as_list()
        assert m_s[-1] == w_s[0]
        mat_r = tf.reshape(mat, shape=[-1, m_s[-1]])  # (b*t,h)
        mul = tf.matmul(mat_r, weight)
        return tf.reshape(mul, shape=[m_s[0], m_s[1], w_s[-1]])

    def __init__(self):
        self.opts = json.load(open("model/config.json"))
        opts = self.opts


        # self.embedding = nn.Embedding(vocab_size + 1, embedding_dim=embedding_size)
        self.emb = tf.Variable(tf.truncated_normal(shape=[98745 + 1, opts["embedding_size"]], mean=0.0, stddev=1.0,dtype=tf.float32))

        # self.a_attention = nn.Linear(embedding_size, 1, bias=False)
        stdv = 1. / math.sqrt(opts["embedding_size"])
        self.a_attention = tf.Variable(tf.random_uniform(minval=-stdv,maxval=stdv,shape=[opts["embedding_size"],1]))

        # Concat Attention
        # self.Wc1 = nn.Linear(2 * encoder_size, encoder_size, bias=False)
        self.Wc1 = tf.Variable(tf.random_uniform(minval=-stdv,maxval=stdv,shape=[2*opts["hidden_size"],opts["hidden_size"]]))
        # self.Wc2 = nn.Linear(2 * encoder_size, encoder_size, bias=False)
        self.Wc2 = tf.Variable( tf.random_uniform(minval=-stdv, maxval=stdv, shape=[2 * opts["hidden_size"], opts["hidden_size"]]))
        # self.vc = nn.Linear(encoder_size, 1, bias=False)
        self.vc = tf.Variable(tf.random_uniform(minval=-stdv, maxval=stdv, shape=[opts["hidden_size"], 1]))

        # Bilinear Attention
        # self.Wb = nn.Linear(2 * encoder_size, 2 * encoder_size, bias=False)
        self.Wb=  tf.Variable(tf.random_uniform(minval=-stdv,maxval=stdv,shape=[2*opts["hidden_size"],2*opts["hidden_size"]]))



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
        with tf.variable_scope("a_encoding"):
            print("a encoding...")
            cell_fw_a = tf.nn.rnn_cell.GRUCell(num_units=opts["hidden_size"]/2)
            cell_bw_a = tf.nn.rnn_cell.GRUCell(num_units=opts["hidden_size"]/2)
            a_embeddings_r=tf.reshape(a_embeddings,shape=[-1,opts["alt_len"],opts["embedding_size"]]) # (3b,a,d)
            a_encoder,_=tf.nn.bidirectional_dynamic_rnn(cell_fw_a,cell_bw_a,a_embeddings_r,dtype=tf.float32)
            a_encoder=tf.concat(a_encoder,axis=2) # (3b,a,h)
            print("a_encoder: {}".format(a_encoder))
            a_score = tf.nn.softmax(self.weight_matmul(a_encoder,self.a_attention),axis=1) # (3b,a,1)
            a_output = tf.matmul(tf.transpose(a_score,perm=[0,2,1]),a_encoder) # (3b,1,a) b* (3b,a,h) -> (3b,1,h)
            a_output = tf.squeeze(a_output) # (3b,h)
            a_embedding = tf.reshape(a_output,shape=[-1,3,opts["hidden_size"]]) # (b,3,h)
            print("a_embedding: {}".format(a_embedding))

        with tf.variable_scope("qp_encoding"):
            print("pp encoding...")
            cell_fw=tf.nn.rnn_cell.GRUCell(num_units=opts["hidden_size"])
            cell_bw=tf.nn.rnn_cell.GRUCell(num_units=opts["hidden_size"])

            # hq, _ = self.q_encoder(p_embedding)
            # hq = F.dropout(hq, self.drop_out)
            hq, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, q_embedding, dtype=tf.float32)  # (b,q,2h)
            hq = tf.concat(hq, axis=2)
            hq=tf.nn.dropout(hq,keep_prob=opts["dropout"])
            # hp, _ = self.p_encoder(q_embedding)
            # hp = F.dropout(hp, self.drop_out)
            hp,_=tf.nn.bidirectional_dynamic_rnn(cell_fw,cell_bw,p_embedding,dtype=tf.float32) # (b,p,2h)
            hp=tf.concat(hp,axis=2)
            hp = tf.nn.dropout(hp, keep_prob=opts["dropout"])
            print("p/q_encoder: {}".format(hq))

        with tf.variable_scope("multiway_attention"):
            print("multi-way attention layer")
            # (1) concatenate
            # _s1 = self.Wc1(hq).unsqueeze(1)
            _s1=tf.expand_dims(self.weight_matmul(hq,self.Wc1),axis=1) # (b,1,q,h)
            # _s2 = self.Wc2(hp).unsqueeze(2)
            _s2=tf.expand_dims(self.weight_matmul(hp,self.Wc2),axis=2) # (b,p,1,h)
            # sjt = self.vc(torch.tanh(_s1 + _s2)).squeeze()
            tanh=tf.reshape(tf.nn.tanh(_s1+_s2),shape=[opts["batch"]*opts["p_len"],-1,opts["hidden_size"]]) # (bp,q,h)
            sjt=tf.squeeze(self.weight_matmul(tanh,self.vc)) # (bp,q,1) -> (bp,q)
            sjt=tf.reshape(sjt,shape=[opts["batch"],opts["p_len"],-1]) # (b,p,q)
            # ait = F.softmax(sjt, 2)
            ait=tf.nn.softmax(sjt,axis=2)
            # qtc = ait.bmm(hq)
            qtc = tf.matmul(ait,hq) # (b,p,q) b* (b,q,2h) -> (b,p,2h)
            print("qtc: {}".format(qtc))

            # (2) bi-linear
            # _s1 = self.Wb(hq).transpose(2, 1)
            _s1=tf.transpose(self.weight_matmul(hq,self.Wb),perm=[0,2,1]) # (b,q,h) -> (b,h,q)
            # sjt = hp.bmm(_s1)
            sjt= tf.matmul(hp,_s1) # (b,p,h) b* (b,h,q) -> (b,p,q)
            # ait = F.softmax(sjt, 2)
            ait = tf.nn.softmax(sjt,2)
            # qtb = ait.bmm(hq)
            qtb = tf.matmul(ait,hq) # (b,p,q) b* (b,q,2h) -> (n,p,2h)



