import json
import math

import tensorflow as tf



class MwAN:
    def random_weight(self, dim_in, dim_out, name=None, stddev=1.0):
        return tf.Variable(tf.truncated_normal([dim_in, dim_out], stddev=stddev / math.sqrt(float(dim_in))), name=name)

    def DropoutWrappedLSTMCell(self, hidden_size, in_keep_prob, name=None):
        cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_size, forget_bias=1.0, state_is_tuple=True, name=name)
        cell = tf.nn.rnn_cell.DropoutWrapper(cell, input_keep_prob=in_keep_prob)
        return cell

    def mat_weight_mul(self, mat, weight):
        """ 带batch的矩阵相乘"""
        # mat(b,n,m) *weight(m,p)
        # => (b,n,m) -> (b*n,m)
        # => (b*n,m)*(m,p)=(b*n,p)
        # return (b,n,p)
        mat_shape = mat.get_shape().as_list()
        weight_shape = weight.get_shape().as_list()
        assert (mat_shape[-1] == weight_shape[0])  # 检查矩阵是否可以相乘
        mat_reshape = tf.reshape(mat, shape=[-1, mat_shape[-1]])  # [b*n,m]
        mul = tf.matmul(mat_reshape, weight)  # [b*n,p]
        return tf.reshape(mul, shape=[-1, mat_shape[1], weight_shape[-1]])  # [b,n,p]

    def random_bias(self, dim, name=None):
        return tf.Variable(tf.truncated_normal(shape=[dim]), name=name)

    def __init__(self):
        self.opts=json.load(open("model/config.json"))
        opts=self.opts
        self.embedding_matrix=self.random_weight(dim_in=opts["vocab_size"],dim_out=opts["embedding_size"])

        # multi way attention weigth
        # concatenate
        self.W_c_p = self.random_weight(dim_in=2 * opts["hidden_size"], dim_out=opts["hidden_size"],
                                        name="concat_att_w_p")
        self.W_c_q = self.random_weight(dim_in=2 * opts["hidden_size"], dim_out=opts["hidden_size"],
                                        name="concat_att_w_q")
        self.V_c = self.random_bias(dim=opts["hidden_size"], name="concat_att_v")
        # bilinear
        self.W_b = self.random_weight(dim_in=2*opts["hidden_size"],dim_out=2*opts["hidden_size"],name="bilinear_att_w")
        # dot
        self.W_d=self.random_weight(dim_in=2*opts["hidden_size"],dim_out=opts["hidden_size"],name="dot_att_w")
        self.V_d=self.random_bias(dim=opts["hidden_size"],name="dot_att_v")
        # minus
        self.W_m=self.random_weight(dim_in=2*opts["hidden_size"],dim_out=opts["hidden_size"],name="minus_att_w")
        self.V_m=self.random_bias(dim=opts["hidden_size"],name="minus_att_v")
        # self att
        self.W_s=self.random_weight(dim_in=2*opts["hidden_size"],dim_out=opts["hidden_size"],name="self_att_w")
        self.V_s = self.random_bias(dim=opts["hidden_size"], name="minus_att_s")

        # prediction layer
        self.W_q=self.random_weight(dim_in=2*opts["hidden_size"],dim_out=opts["hidden_size"],name="prediction_layer_att_w_q")
        self.V_q=self.random_bias(dim=opts["hidden_size"],name="prediction_layer_att_v_q")
        self.W_p1=self.random_weight(dim_in=2*opts["hidden_size"],dim_out=opts["hidden_size"],name="prediction_layer_att_w_p1")
        self.W_p2=self.random_weight(dim_in=2*opts["hidden_size"],dim_out=opts["hidden_size"],name="prediction_layer_att_w_p2")
        self.V_p = self.random_bias(dim=opts["hidden_size"], name="prediction_layer_att_v_p")
        self.W_predict=self.random_weight(dim_in=2*opts["hidden_size"],dim_out=opts["embedding_size"])

        # answer attention
        self.V_a=self.random_bias(dim=opts["hidden_size"],name="answer_att_v")

        # gate for aggregating
        self.W_g = self.random_weight(dim_in=4*opts["hidden_size"],dim_out=4*opts["hidden_size"],name="aggregate_gate")

        # aggregate attention
        self.W_agg = self.random_weight(dim_in=4* opts["hidden_size"], dim_out=opts["hidden_size"],name="aggregate_att_w")
        self.V_agg = self.random_bias(dim= opts["hidden_size"],name="aggregate_att_b")


    def build(self):
        print("building model...")
        opts=self.opts

        # placeholder
        #FIXME: batch should be changed to None
        query=tf.placeholder(dtype=tf.int32,shape=[opts["batch"],opts["q_len"]])
        para =tf.placeholder(dtype=tf.int32,shape=[opts["batch"],opts["p_len"]])
        ans  =tf.placeholder(dtype=tf.int32,shape=[opts["batch"],3,opts["alt_len"]]) # 每个ans中有三个小句，第一句为正确答案 FIXME: alt_len should be None

        # embedding
        with tf.variable_scope("Embedding_Encoding_Layer"):
            print("Layer1: Embedding& Encoding Layer")
            q_emb=tf.nn.embedding_lookup(self.embedding_matrix,query) # (b,q,emb)
            p_emb=tf.nn.embedding_lookup(self.embedding_matrix,para) # (b,p,emb)
            a_emb=tf.nn.embedding_lookup(self.embedding_matrix,ans) # (b,3,a,emb)

            print("p/q_emb:",q_emb)

            q_emb_us=tf.unstack(q_emb,axis=1) # (q,b,emb)
            p_emb_us=tf.unstack(p_emb,axis=1) # (p,b,emb)

            fw_cells=[self.DropoutWrappedLSTMCell(hidden_size=opts["hidden_size"],in_keep_prob=opts["dropout"]) for _ in range(2)]
            bw_cells=[self.DropoutWrappedLSTMCell(hidden_size=opts["hidden_size"],in_keep_prob=opts["dropout"]) for _ in range(2)]
            print("encoding q...")
            h_q,_,_=tf.contrib.rnn.stack_bidirectional_rnn(fw_cells, bw_cells, q_emb_us, dtype=tf.float32, scope="q_encoding")
            print("encoding p...")
            h_p,_,_=tf.contrib.rnn.stack_bidirectional_rnn(fw_cells, bw_cells, p_emb_us, dtype=tf.float32, scope="p_encoding")

            h_q=tf.nn.dropout(tf.stack(h_q,axis=1),keep_prob=opts["dropout"])
            h_p=tf.nn.dropout(tf.stack(h_p,axis=1),keep_prob=opts["dropout"])

        print("p/q_enc:",h_q)

        with tf.variable_scope("Multiway_Matching_Layer"):
            print("Layer2: Multi-way Matching Layer")

            # Concat Attention
            print("obtaining concat attention...")
            """adapted from pytorch
           _s1 = self.Wc1(hq).unsqueeze(1) # (b,q,2h) * (2h,h) = (b,p,h) =us1= (b,1,q,h)
           _s2 = self.Wc2(hp).unsqueeze(2) # (b,p,2h) * (2h,h) = (b,p,h) =us2= (b,p,1,h)
           sjt = self.vc(torch.tanh(_s1 + _s2)).squeeze() # 自动广播(2,3维度) (b,p,q,h) * (h,1) = (b,p,q)
           ait = F.softmax(sjt, 2) # (b,p,q)
           qtc = ait.bmm(hq) # (b,p,q) b* (b,q,2h) = (b,p,2h)
           """
            _s1=self.mat_weight_mul(h_q, self.W_c_q)
            _s1=tf.expand_dims(_s1,axis=1) # (b,1,q,h)
            _s2=self.mat_weight_mul(h_p, self.W_c_p)
            _s2=tf.expand_dims(_s2,axis=2) # (b,p,1,h)
            tanh=tf.tanh(_s1+_s2) # (b,p,q,h) 在维度为1的位置上自动广播 相当于til操作

            # sjt=tf.squeeze(tf.matmul(tanh,tf.reshape(self.V_c,shape=[-1,1]))) # (b,p,q,h) * (h,1) =sq=> (b,p,q) TODO: tf 中(b,p,q,h) * (h,1) 不可直接matmul
            sjt=tf.matmul(tf.reshape(tanh,[-1,opts["hidden_size"]]),tf.reshape(self.V_c,[-1,1]))# (b*p*q,h) * (h,1) => (b*p*q,1)
            sjt=tf.squeeze(tf.reshape(sjt,shape=[opts["batch"],opts["p_len"],opts["q_len"],-1])) # (b,p,q,1) =sq=> (b,p,q)
            ait=tf.nn.softmax(sjt,axis=2) # (b,p,q)

            # apply attention weight
            qtc= tf.matmul(ait,h_q) # (b,p,q) batch* (b,q,2h) => (b,p,2h) 当识别到有batch时 tf.matmul 自动转变为keras.K.batch_dot

            print("_s1: {} | _s2: {}".format(_s1, _s2))
            print("tanh:", tanh, "自动广播")
            print("sjt: {}".format(sjt))
            print("qtc: {}".format(qtc))
            # Bi-linear Attention
            print("obtaining bi-linear attention...")
            """adapted from pytorch
           _s1 = self.Wb(hq).transpose(2, 1) # (b,q,2h) * (2h,2h) =trans= (b,2h,q)
           sjt = hp.bmm(_s1) # (b,p,2h) b* (b,2h,q) = (b,p,q)
           ait = F.softmax(sjt, 2) # (b,p,q)
           qtb = ait.bmm(hq) # (b,p,q) b* (b,q,2h) = (b,p,2h)
           """
            _s = self.mat_weight_mul(h_q, self.W_b) # (b,q,2h) * (2h,2h) => (b,q,2h)
            _s = tf.transpose(_s,perm=[0,2,1]) # (b,q,2h) => (b,2h,q)
            sjt= tf.matmul(h_p,_s) # (b,p,2h) batch* (b,2h,q) => (b,p,q)
            ait=tf.nn.softmax(sjt,axis=2) # (b,p,q)
            qtb=tf.matmul(ait,h_q) # (b,p,q) batch* (b,q,2h) => (b,p,2h) 顺序不能反

            print("qtb: {}".format(qtb))

            # Dot Attention
            print("obtaining dot attention...")
            """ adapted from pytorch
           _s1 = hq.unsqueeze(1) # (b,q,2h) =us1= (b,1,q,2h)
           _s2 = hp.unsqueeze(2) # (b,p,2h) =us2= (b,p,1,2h)
           sjt = self.vd(torch.tanh(self.Wd(_s1 * _s2))).squeeze() # (b,p,q,2h)*(2h,h)*(h,) =sq= (b,p,q)
           ait = F.softmax(sjt, 2) # (b,p,q)
           qtd = ait.bmm(hq)  # (b,p,q) b* (b,q,2h) = (b,p,2h)
           """
            _s1 = tf.expand_dims(h_q,axis=1) # (b,q,2h) => (b,1,q,2h)
            _s2 = tf.expand_dims(h_p,axis=2) # (b,p,2h) => (b,p,1,2h)

            _tanh=tf.tanh(self.mat_weight_mul(_s1 * _s2, self.W_d)) # 乘法自动广播 (b,p,q,2h)*(2h,h) =mat_weight_mul=> (b,p*q,h)
            tanh=tf.reshape(_tanh,[opts["batch"],opts["p_len"],opts["q_len"],-1]) # (b,p,q,h)
            _sjt=self.mat_weight_mul(_tanh,tf.reshape(self.V_d,[-1,1])) # (b,p*q,1) =sq=> (b,p*q,1)
            sjt=tf.squeeze(tf.reshape(_sjt,[opts["batch"],opts["p_len"],opts["q_len"],-1])) # (b.p,q)
            ait=tf.nn.softmax(sjt,axis=2)

            qtd = tf.matmul(ait,h_q) # (b,p,q) batch* (b,q,2h) => (b,p,2h)
            print("_tanh from mat_weight_mul: {}".format(_tanh))
            print("tanh reshaped: {}".format(tanh))
            print("_sjt from mat_weight_mul: {} ".format(_sjt))
            print("sjt reshaped: {}".format(sjt))
            print("qtd: {}".format(qtd))

            # Minus Attention
            print("obtaining minus attention...")
            """adapted from pytorch
           _s1 = hq.unsqueeze(1) # (b,1,q,2h)
           _s2 = hp.unsqueeze(2) # (b,p,1,2h)
           sjt = self.vm(torch.tanh(self.Wm(_s1 - _s2))).squeeze() # (b,p,q,2h)*(2h,h)(h,) =sq= (b,p,q)
           ait = F.softmax(sjt, 2) # (b,p,q)
           qtm = ait.bmm(hq) # (b,p,q) b* (b,q,2h) = (b,p,2h)
           """
            _s1 = tf.expand_dims(h_q, axis=1)  # (b,q,2h) => (b,1,q,2h)
            _s2 = tf.expand_dims(h_p, axis=2)  # (b,p,2h) => (b,p,1,2h)

            _tanh=tf.tanh(self.mat_weight_mul(_s1-_s2,self.W_m)) # (b,p*q,h)
            _sjt=self.mat_weight_mul(_tanh,tf.reshape(self.V_m,[-1,1])) # (b,p*q,1)
            sjt=tf.squeeze(tf.reshape(_sjt,[opts["batch"],opts["p_len"],opts["q_len"],-1])) # (b,p,q)
            ait=tf.nn.softmax(sjt,axis=2)
            qtm=tf.matmul(ait,h_q) # (b,p,q) batch* (b,q,2h) => (n,p,2h)

            print("qtm: {}".format(qtm))

            # Self Matching Attention
            print("obtaining self attention...")
            """adapted from pytorch
           _s1 = hp.unsqueeze(1) # (b,1,p,2h)
           _s2 = hp.unsqueeze(2) # (b,p,1,2h)
           sjt = self.vs(torch.tanh(self.Ws(_s1 * _s2))).squeeze() # (b,p,p,2h)*(2h,h)*(h,) =sq= (b,p,p)
           ait = F.softmax(sjt, 2) # (b,p,p)
           qts = ait.bmm(hp) # (b,p,p) b* (b,p,2h) = (b,p,2h)
           """
            _s1=tf.expand_dims(h_p,axis=1) # (b,1,p,2h)
            _s2=tf.expand_dims(h_p,axis=2) # (b,p,1,2h)
            tanh=tf.tanh(self.mat_weight_mul(_s1*_s2,self.W_s)) # (b,p*p,h)
            sjt=self.mat_weight_mul(tanh,tf.reshape(self.V_s,[-1,1])) # (b,p*p,1)
            sjt=tf.squeeze(tf.reshape(sjt,[opts["batch"],opts["p_len"],opts["p_len"],-1])) # (b,p,p)
            ait=tf.nn.softmax(sjt,axis=2)
            qts=tf.matmul(ait,h_p) # (b,p,p) batch* (b,p,2h) => (b,p,2h)
            print("qts: {}".format(qts))

        with tf.variable_scope("Aggregate_Layer"):
            print("Layer3: Aggregate Layer")

            def get_x(qt,scope):
                print("adding {} ...".format(scope),end=" ")
                # 公式8a-8e
                with tf.variable_scope(scope):
                    xt=tf.concat([qt,h_p],axis=2) # (b,p,4h)
                    gt=tf.sigmoid(self.mat_weight_mul(xt,self.W_g)) # (b,p,4h) (4h,4h)
                    xt_star=tf.squeeze(tf.multiply(xt,gt)) # ( b,p,4h)
                    # FIXME: 计算能力提升后再添加此层
                    # fc=[self.DropoutWrappedLSTMCell(hidden_size=opts["hidden_size"],in_keep_prob=opts["dropout"]) for _ in range(2)]
                    # bc=[self.DropoutWrappedLSTMCell(hidden_size=opts["hidden_size"],in_keep_prob=opts["dropout"]) for _ in range(2)]
                    #
                    # xt_star_unstack=tf.unstack(xt_star,axis=1)
                    # ht,_,_=tf.contrib.rnn.stack_bidirectional_rnn(fc, bc, xt_star_unstack, dtype=tf.float32,
                    #                                        scope="gating")
                    # ht=tf.stack(ht,axis=1)
                    ht=xt_star
                    ht=tf.nn.dropout(ht,opts["dropout"]) # ( b,p,4h)
                print("shape: {}".format(ht))
                return ht
            htc=get_x(qtc,"c_gate")
            htb=get_x(qtb,"b_gate")
            htd=get_x(qtd,"d_gate")
            htm=get_x(qtm,"m_gate")
            hts=get_x(qts,"s_gate") # (b,p,2h)
            aggre=tf.concat([hts,htc,htb,htd,htm],axis=2) # (b,p,10h)
            aggre_reshape=tf.reshape(aggre,shape=[opts["batch"],opts["p_len"]*5,-1]) # (b,p*5,2h)
            print("aggre_reshaped: {}".format(aggre_reshape))

            # 公式9a-9c
            tanh=tf.tanh(self.mat_weight_mul(aggre_reshape,self.W_agg))# (b,p*5,h)
            sj=self.mat_weight_mul(tanh,tf.reshape(self.V_agg,shape=[-1,1])) # (b,p*5,1)
            # print(sj)
            sj=tf.reshape(sj,[opts["batch"],opts["p_len"],5,-1])
            sj=tf.transpose(sj,perm=[0,1,3,2])  # (b,p,1,5)
            aj=tf.nn.softmax(sj,axis=3)
            aj=tf.reshape(aj,[opts["batch"],opts["p_len"],5,-1])
            xt=tf.matmul(tf.reshape(aj,[opts["batch"]*opts["p_len"],-1,5]),tf.reshape(aggre_reshape,[opts["batch"]*opts["p_len"],5,-1])) # (b*p,1,5)*(b*p,5,2h) => (b*p,1,2h)
            print("xt: {}".format(xt))
            xt=tf.reshape(xt,[opts["batch"],opts["p_len"],1,4*opts["hidden_size"]])
            xt=tf.squeeze(xt) # (b,p,4h)
            print("xt: {}".format(xt))

            # aggregate=tf.concat([h_p, qts, qtc, qtd, qtb, qtm],axis=2) # (b,p,12h)
            # print("aggregate: {}".format(aggregate))
            xt_unstack=tf.unstack(xt,axis=1) # (p,b,12h)

            fw_cells = [self.DropoutWrappedLSTMCell(hidden_size=opts["hidden_size"], in_keep_prob=opts["dropout"]) for _
                        in range(2)]
            bw_cells = [self.DropoutWrappedLSTMCell(hidden_size=opts["hidden_size"], in_keep_prob=opts["dropout"]) for _
                        in range(2)]
            aggregate_representation,_,_=tf.contrib.rnn.stack_bidirectional_rnn(fw_cells, bw_cells,xt_unstack, dtype=tf.float32, scope="aggregate_representation")

            aggregate_representation=tf.stack(aggregate_representation,axis=1) # (b,p,2h) bi-directional
            aggregate_representation=tf.nn.dropout(aggregate_representation,opts["dropout"])
            print("aggregate_rep: {}".format(aggregate_representation))

        with tf.variable_scope("Prediction_Layer"):
            print("Layer4: Prediction Layer")
            # 公式：11a-11c
            tanh = tf.tanh(self.mat_weight_mul(h_q,self.W_q))  # (b,q,h)
            sj = self.mat_weight_mul(tanh,tf.reshape(self.V_q,[-1,1]))  # (b,q,1)
            print("sjt: {}".format(sj))
            sj_trans = tf.transpose(sj,perm=[0,2,1])  # (b,1,q)为使用batch_dot 计算attention需要先转置
            ai=tf.nn.softmax(sj_trans,axis=2)
            r_q= tf.matmul(ai,h_q) # (b,1,q)*(b,q,2h) => (b,1,2h)
            print("r_q:",r_q)
            # 公式：12a-12c
            tanh=tf.tanh(self.mat_weight_mul(aggregate_representation,self.W_p1)+
                         self.mat_weight_mul(r_q,self.W_p2)) # (b,p,2h)*(2h,h) => (b,p,h)
            sj = self.mat_weight_mul(tanh,tf.reshape(self.V_p,[-1,1]))  # (b,p,1)
            sj_trans = tf.transpose(sj,perm=[0,2,1]) # (b,1,p)矩阵计算机attention 使用batch_dot 需要先转置
            ai=tf.nn.softmax(sj_trans,axis=2)
            r_p=tf.matmul(ai,aggregate_representation) # (b,1,p) batch* (b,p,2h)

            # output
            encoder_out = tf.nn.relu(self.mat_weight_mul(r_p,self.W_predict)) # (b,p,2h)*(2h,h)=>(b,1,h)
            encoder_out = tf.nn.dropout(encoder_out,opts["dropout"])
            print("encoder_out: {}".format(encoder_out))

            # 处理answer的embedding
            a_emb_rs = tf.reshape(a_emb, shape=[opts["batch"] * 3, -1, opts["embedding_size"]])  # (3b,a,emb)
            a_emb_rs_us = tf.unstack(a_emb_rs, axis=1)  # (a,3b,emb)
            print("encoding a...")
            a_fw_cells = [self.DropoutWrappedLSTMCell(hidden_size=opts["hidden_size"] / 2, in_keep_prob=opts["dropout"])
                          for _
                          in range(2)]
            a_bw_cells = [self.DropoutWrappedLSTMCell(hidden_size=opts["hidden_size"] / 2, in_keep_prob=opts["dropout"])
                          for _
                          in range(2)]
            a_enc, _, _ = tf.contrib.rnn.stack_bidirectional_rnn(a_fw_cells, a_bw_cells, a_emb_rs_us, dtype=tf.float32,
                                                                 scope="a_encoding")  # a_cell 要短些（后续计算需要）
            a_enc = tf.nn.dropout(tf.stack(a_enc, axis=1), keep_prob=opts["dropout"])  # (3b,a,h)
            print("a_enc: {}".format(a_enc))
            a_sj = self.mat_weight_mul(a_enc,tf.reshape(self.V_a,[-1,1])) # (3b,a,h)*(h,1)=> (3b,a,1)
            a_ai = tf.nn.softmax(a_sj,axis=1) # (3b,a,1)
            a_ai= tf.transpose(a_ai,perm=[0,2,1]) # (3b,1,a) attention 转置
            a_output=tf.matmul(a_ai,a_enc) # (3b,1,a)*(3b,a,h) => (3b,1,h)
            a_output=tf.squeeze(a_output) # (3b,h)
            a_output=tf.reshape(a_output,[opts["batch"],3,-1]) # (b,3,h)
            # 结合encoder_out与answer
            score = tf.matmul(a_output,tf.transpose(encoder_out,perm=[0,2,1])) # (b,3,h) batch* (b,h,1) => (b,3,1)
            score = tf.nn.softmax(tf.squeeze(score),axis=1,name="prediction") # (b,3)
            print("socre: {}".format(score))


            print("complying...")
            print("loss...")
            loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf.constant([[1,0,0] for _ in range(opts["batch"])]),logits=score))
            print("optimizer...")
            optimizer = tf.train.AdamOptimizer().minimize(loss)
            print("predict..")
            predict=tf.argmax(score)
            print("dict...")

            test_q=tf.squeeze(query[:,0])
            test_p=tf.squeeze(para[:,0])
            test_a=tf.squeeze(ans[:,0,0])
            test_op=[test_p,test_a,test_q]
            print("test_op(用于快速测试feed_dict): {}".format(test_op))

            tensor_dict={
                "q" : query,
                "p" : para,
                "a" : ans
            }
            return  loss,optimizer,predict,tensor_dict,test_op


