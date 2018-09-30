import math

import tensorflow as tf
import json
import tensorflow.contrib as tc
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
        self.opts = json.load(open("model/aichallenger_config.json"))
        opts = self.opts


        # self.embedding = nn.Embedding(vocab_size + 1, embedding_dim=embedding_size)
        self.emb = tf.Variable(tf.truncated_normal(shape=[98745 + 1, opts["embedding_size"]], mean=0.0, stddev=1.0,dtype=tf.float32))

        # self.a_attention = nn.Linear(embedding_size, 1, bias=False)
        stdv = 1. / math.sqrt(opts["embedding_size"])
        self.a_attention = tf.Variable(tf.random_uniform(minval=-stdv, maxval=stdv, shape=[opts["embedding_size"], 1]))

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

        # Dot Attention :
        # self.Wd = nn.Linear(2 * encoder_size, encoder_size, bias=False)
        self.Wd = tf.Variable(tf.random_uniform(minval=-stdv,maxval=stdv,shape=[2*opts["hidden_size"],opts["hidden_size"]]))
        # self.vd = nn.Linear(encoder_size, 1, bias=False)
        self.vd = tf.Variable(tf.random_uniform(minval=-stdv,maxval=stdv,shape=[opts["hidden_size"],1]))

        # Minus Attention :
        # self.Wm = nn.Linear(2 * encoder_size, encoder_size, bias=False)
        self.Wm = tf.Variable(tf.random_uniform(minval=-stdv,maxval=stdv,shape=[2*opts["hidden_size"],opts["hidden_size"]]))
        # self.vm = nn.Linear(encoder_size, 1, bias=False)
        self.vm = tf.Variable(tf.random_uniform(minval=-stdv,maxval=stdv,shape=[opts["hidden_size"],1]))

        # self.Ws = nn.Linear(2 * encoder_size, encoder_size, bias=False)
        self.Ws = tf.Variable(tf.random_uniform(minval=-stdv,maxval=stdv,shape=[2*opts["hidden_size"],opts["hidden_size"]]))
        # self.vs = nn.Linear(encoder_size, 1, bias=False)
        self.vs= tf.Variable(tf.random_uniform(minval=-stdv,maxval=stdv,shape=[opts["hidden_size"],1]))

        """
        prediction layer
       """
        # self.Wq = nn.Linear(2 * encoder_size, encoder_size, bias=False)
        self.Wq=tf.Variable(tf.random_uniform(minval=-stdv,maxval=stdv,shape=[2*opts["hidden_size"],opts["hidden_size"]]))
        # self.vq = nn.Linear(encoder_size, 1, bias=False)
        self.vq =tf.Variable(tf.random_uniform(minval=-stdv,maxval=stdv,shape=[opts["hidden_size"],1]))
        # self.Wp1 = nn.Linear(2 * encoder_size, encoder_size, bias=False)
        self.Wp1 =tf.Variable(tf.random_uniform(minval=-stdv,maxval=stdv,shape=[2*opts["hidden_size"],opts["hidden_size"]]))
        # self.Wp2 = nn.Linear(2 * encoder_size, encoder_size, bias=False)
        self.Wp2 =tf.Variable(tf.random_uniform(minval=-stdv,maxval=stdv,shape=[2*opts["hidden_size"],opts["hidden_size"]]))
        # self.vp = nn.Linear(encoder_size, 1, bias=False)
        self.vp =tf.Variable(tf.random_uniform(minval=-stdv,maxval=stdv,shape=[opts["hidden_size"],1]))
        # self.prediction = nn.Linear(2 * encoder_size, embedding_size, bias=False)
        self.prediction =tf.Variable(tf.random_uniform(minval=-stdv,maxval=stdv,shape=[2*opts["hidden_size"],opts["embedding_size"]]))

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
            cell_fw_a = tc.rnn.MultiRNNCell([tc.rnn.GRUCell(num_units=opts["hidden_size"]/2) for _ in range(1)])
            cell_bw_a = tc.rnn.MultiRNNCell([tc.rnn.GRUCell(num_units=opts["hidden_size"]/2) for _ in range(1)])
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
            cell_fw=tc.rnn.MultiRNNCell([tc.rnn.GRUCell(num_units=opts["hidden_size"]) for _ in range(1)])
            cell_bw=tc.rnn.MultiRNNCell([tc.rnn.GRUCell(num_units=opts["hidden_size"]) for _ in range(1)])

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
            print("qtb: {}".format(qtb))

            # （3）dot
            # _s1 = hq.unsqueeze(1)
            _s1=tf.expand_dims(hq,axis=1) # (b,1,q,2h)
            # _s2 = hp.unsqueeze(2)
            _s2=tf.expand_dims(hp,axis=2) # (b,p,1,2h)
            # sjt = self.vd(torch.tanh(self.Wd(_s1 * _s2))).squeeze()
            dot=tf.reshape(_s1*_s2,shape=[opts["batch"]*opts["p_len"],-1,2*opts["hidden_size"]]) # (b*p,q,2h)
            tanh=tf.tanh(self.weight_matmul(dot,self.Wd)) # (b*p,q,h)
            sjt=tf.reshape(self.weight_matmul(tanh,self.vd),shape=[opts["batch"],opts["p_len"],opts["q_len"],1]) #
            # ait = F.softmax(sjt, 2)
            ait=tf.squeeze(tf.nn.softmax(sjt,axis=2)) # (b,p,q)
            # qtd = ait.bmm(hq)
            qtd=tf.matmul(ait,hq) # (b,p,2h)
            print("qtd: {}".format(qtd))

            # (4) minus
            # sjt = self.vm(torch.tanh(self.Wm(_s1 - _s2))).squeeze()
            minus=tf.reshape(_s1-_s2,[opts["batch"]*opts["p_len"],-1,2*opts["hidden_size"]]) # (b*p,q,2h)
            tanh=tf.tanh(self.weight_matmul(minus,self.Wm))
            sjt=self.weight_matmul(tanh,self.vc) # (b,p,q,1)
            # ait = F.softmax(sjt, 2)
            ait=tf.squeeze(tf.nn.softmax(ait,axis=2))
            # qtm = ait.bmm(hq)
            qtm=tf.matmul(ait,hq) # (b,p,q) (b,q,h) -> (b.p,2h)
            print("qtm: {}".format(qtm))

            # (5) self attention
            # _s1 = hp.unsqueeze(1)
            _s1=tf.expand_dims(hp,axis=1)
            # _s2 = hp.unsqueeze(2)
            _s2=tf.expand_dims(hp,axis=2)
            # sjt = self.vs(torch.tanh(self.Ws(_s1 * _s2))).squeeze()
            sel=tf.reshape(_s1*_s2,shape=[opts["batch"]*opts["p_len"],opts["p_len"],-1]) # (b*p,p,2h)
            tanh=tf.tanh(self.weight_matmul(sel,self.Ws))
            sjt=self.weight_matmul(tanh,self.vs)
            sjt=tf.reshape(sjt,shape=[opts["batch"],opts["p_len"],opts["p_len"],-1])
            # ait = F.softmax(sjt, 2)
            ait=tf.squeeze(tf.nn.softmax(sjt,axis=2)) # (b,p,p)
            # qts = ait.bmm(hp)
            qts= tf.matmul(ait,hp)
            print("qts: {}".format(qts))

        with tf.variable_scope("aggregating_layer"):
            print("aggregating layer:")
            # aggregation = torch.cat([hp, qts, qtc, qtd, qtb, qtm], 2)
            aggregation=tf.concat([hp,qts,qts,qtd,qtb,qtm],2)
            # self.gru_agg = nn.GRU(12 * encoder_size, encoder_size, batch_first=True, bidirectional=True)
            fw_cell=tc.rnn.MultiRNNCell([tc.rnn.GRUCell(num_units=opts["hidden_size"]) for _ in range(1)])
            bw_cell=tc.rnn.MultiRNNCell([tc.rnn.GRUCell(num_units=opts["hidden_size"]) for _ in range(1)])
            # aggregation_representation, _ = self.gru_agg(aggregation)
            aggregation_representation,_ =tf.nn.bidirectional_dynamic_rnn(fw_cell,bw_cell,aggregation,dtype=tf.float32)
            aggregation_representation=tf.concat(aggregation_representation,axis=2)
            print("aggregation_representaion: {}".format(aggregation_representation))

        with tf.variable_scope("prediction_layer"):
            print("prediction layer:")
            # sj = self.vq(torch.tanh(self.Wq(hq))).transpose(2, 1)
            tanh=tf.nn.tanh(self.weight_matmul(hq,self.Wq))
            sj=tf.transpose(self.weight_matmul(tanh,self.vq),perm=[0,2,1])
            # rq = F.softmax(sj, 2).bmm(hq)
            ai=tf.nn.softmax(sj,2) # (b,1,q)
            rq=tf.matmul(ai,hq) # (b,1,q) (b,q,2h) -> (b,1,2h)

            # sj = F.softmax(self.vp(self.Wp1(aggregation_representation) + self.Wp2(rq)).transpose(2, 1), 2)
            add=self.weight_matmul(aggregation_representation,self.Wp1) + self.weight_matmul(rq,self.Wp2)
            sj=tf.transpose(self.weight_matmul(add,self.vp),perm=[0,2,1])
            sj=tf.nn.softmax(sj,2) # (b,1,p)
            # rp = sj.bmm(aggregation_representation)
            rp = tf.matmul(sj,aggregation_representation) # (b,1,p) (b,p,2h) -> (b,1,2h)
            print("rp: {}".format(rp))
            # encoder_output = F.dropout(F.leaky_relu(self.prediction(rp)), self.drop_out)
            full=self.weight_matmul(rp,self.prediction) # (b,1,2h) (2h,h) -> (b,1,h)
            encoder_output = tf.nn.dropout(tf.nn.leaky_relu(full),opts["dropout"]) # (b,1,h)
            # score = F.softmax(a_embedding.bmm(encoder_output.transpose(2, 1)).squeeze(), 1)
            bmm=tf.matmul(a_embedding,tf.transpose(encoder_output,[0,2,1])) # (b,3,h) (b,h,1) -> (b,3,1)
            bmm=tf.squeeze(bmm) # (b,3)
            score=tf.nn.softmax(bmm,axis=1)
            print("socre: {}".format(score))

        print("complying...")
        print("loss...")
        loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf.constant([[1, 0, 0] for _ in range(opts["batch"])]),
                                                       logits=score))
        print("optimizer...")
        optimizer = tf.train.AdamOptimizer(name="opt").minimize(loss)
        print("opt: {}".format(optimizer))
        print("predict..")
        predict = tf.argmax(score, axis=1, name="pred")  # (10,1)
        print("pred: {}".format(predict))
        print("dict...")

        test_q = tf.squeeze(query[:, 0])
        test_p = tf.squeeze(passage[:, 0])
        test_a = tf.squeeze(answer[:, 0, 0])
        test_op = [test_p, test_a, test_q]
        print("test_op(用于快速测试feed_dict): {}".format(test_op))

        tensor_dict = {
            "q": query,
            "p": passage,
            "a": answer
        }
        return loss, optimizer, predict, tensor_dict, test_op