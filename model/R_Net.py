import json
import math

import tensorflow as tf



class R_Net:
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

        # QP_match
        self.W_u_q = self.random_weight(dim_in=2*opts["hidden_size"],dim_out=opts["hidden_size"],name="W_u_q")
        self.W_u_p_t = self.random_weight(dim_in=2*opts["hidden_size"],dim_out=opts["hidden_size"],name="W_u_p_t")
        self.W_v_p_t_L = self.random_weight(dim_in=opts["hidden_size"], dim_out=opts["hidden_size"], name="W_v_p_t_L")
        self.V_qp =self.random_bias(dim=opts["hidden_size"],name="V_qp")
        self.W_qp_gate= self.random_weight(dim_in=4*opts["hidden_size"],dim_out=4*opts["hidden_size"],name="W_qp_gate")

        # QP_match GRU
        with tf.variable_scope("QP_match"):
            self.QPmatch_cell = self.DropoutWrappedLSTMCell(hidden_size=opts["hidden_size"],
                                                            in_keep_prob=opts["dropout"])
            self.QPmatch_state = self.QPmatch_cell.zero_state(batch_size=opts["batch"],  # QP需要提取中间状态所以保留
                                                              dtype=tf.float32)  # RNN单元的初始状态
        #  self Match
        self.W_v_p1=self.random_weight(dim_in=opts["hidden_size"],dim_out=opts["hidden_size"],name="W_v_p1")
        self.W_v_p2=self.random_weight(dim_in=opts["hidden_size"],dim_out=opts["hidden_size"],name="W_v_p2")
        self.V_sm=self.random_bias(dim=opts["hidden_size"],name="V_sm")
        self.W_sm_gate= self.random_weight(dim_in=2*opts["hidden_size"],dim_out=2*opts["hidden_size"],name="W_sm_gate")

        # prediction layer
        self.W_q = self.random_weight(dim_in=2 * opts["hidden_size"], dim_out=opts["hidden_size"],
                                      name="prediction_layer_att_w_q")
        self.V_q = self.random_bias(dim=opts["hidden_size"], name="prediction_layer_att_v_q")
        self.W_p1 = self.random_weight(dim_in=2 * opts["hidden_size"], dim_out=opts["hidden_size"],
                                       name="prediction_layer_att_w_p1")
        self.W_p2 = self.random_weight(dim_in=2 * opts["hidden_size"], dim_out=opts["hidden_size"],
                                       name="prediction_layer_att_w_p2")
        self.V_p = self.random_bias(dim=opts["hidden_size"], name="prediction_layer_att_v_p")
        self.W_predict = self.random_weight(dim_in=2 * opts["hidden_size"], dim_out=opts["embedding_size"])

        # attention for answer encoding
        self.V_a = self.random_bias(dim=opts["hidden_size"],name="V_answer")


    def build(self):
        print("building model...")
        opts=self.opts

        # placeholder
        # FIXME: batch should be changed to None
        query = tf.placeholder(dtype=tf.int32, shape=[opts["batch"], opts["q_len"]], name="query")
        para = tf.placeholder(dtype=tf.int32, shape=[opts["batch"], opts["p_len"]], name="para")
        ans = tf.placeholder(dtype=tf.int32, shape=[opts["batch"], 3, opts["alt_len"]],
                             name="ans")  # 每个ans中有三个小句，第一句为正确答案 FIXME: alt_len should be None

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
            u_q,_,_=tf.contrib.rnn.stack_bidirectional_rnn(fw_cells, bw_cells, q_emb_us, dtype=tf.float32, scope="q_encoding")
            print("encoding p...")
            u_p,_,_=tf.contrib.rnn.stack_bidirectional_rnn(fw_cells, bw_cells, p_emb_us, dtype=tf.float32, scope="p_encoding")

            u_q=tf.nn.dropout(tf.stack(u_q,axis=1),keep_prob=opts["dropout"])
            u_p=tf.nn.dropout(tf.stack(u_p,axis=1),keep_prob=opts["dropout"])

        print("p/q_enc:",u_q)

        with tf.variable_scope("Question_Passage_Attention_Layer"):
            print("Layer2: Question Passage Attention Layer:")
            v_p=[]
            for t in range(opts["p_len"]):
                u_q_W=self.mat_weight_mul(u_q,self.W_u_q) # (b,q,2h)*(2h,h) -> (b,q,h)
                u_p_t=tf.reshape(u_p[:,t,:],[opts["batch"],1,2*opts["hidden_size"]])
                u_p_t_W=self.mat_weight_mul(tf.concat([u_p_t]*opts["q_len"],axis=1),self.W_u_p_t) #(b,q,h)

                if t == 0:
                    tanh=tf.tanh(u_q_W+u_p_t_W)
                else:
                    v_p_t_l=tf.reshape(v_p[t-1],[opts["batch"],1,-1])
                    v_p_t_l_W=self.mat_weight_mul(tf.concat([v_p_t_l]*opts["q_len"],axis=1), self.W_v_p_t_L)  #(b,q,h)

                    tanh=tf.tanh(u_q_W+u_p_t_W+v_p_t_l_W)
                s_t_j=self.mat_weight_mul(tanh,tf.reshape(self.V_qp,[-1,1])) # (b,q,1)
                # print("s_t_j: {}".format(s_t_j))
                a_t_i=tf.nn.softmax(s_t_j,axis=1)
                a_t_i=tf.transpose(a_t_i,perm=[0,2,1]) #(b,1,q)
                c_t = tf.squeeze(tf.matmul(a_t_i,u_q)) # (b,1,q) x (b,q,2h)  -> (b,1,2h) -> (b,2h)
                # print("c_t: {}".format(c_t))

                # gate
                u_p_t_c_t=tf.concat([tf.squeeze(u_p_t),c_t],axis=1) # (b,4h)
                qp_gate=tf.matmul(u_p_t_c_t,self.W_qp_gate) # (b,4h)*(4h,4h) -> (b,4h)
                qp_gate=tf.sigmoid(qp_gate)
                u_p_t_c_t_star=tf.multiply(u_p_t_c_t,qp_gate) # (b,4h)
                # print("[u_p,c_t]* : {}".format(u_p_t_c_t_star))

                # QP_match
                with tf.variable_scope("QP_match"):
                    output, self.QPmatch_state = self.QPmatch_cell(u_p_t_c_t_star,  # (b,4*h)
                                                                   self.QPmatch_state)  # 吧输入 和上一步的状态一起作为输入
                    v_p.append(output)
            v_p = tf.stack(v_p, axis=1)  # (b,p,h) 经过RNN后 形状末端的 4*h又变回h（由QPmatch_cell的隐层决定 ）
            v_p = tf.nn.dropout(v_p, opts["dropout"])
            print('v_P:', v_p)


        with tf.variable_scope("Self_Attention_Layer"):
            print("Layer3: Self Attention Layer:")
            v_p_1_W=tf.matmul(tf.reshape(v_p,[-1,opts["hidden_size"]]),self.W_v_p1)
            v_p_2_W=tf.matmul(tf.reshape(v_p,[-1,opts["hidden_size"]]),self.W_v_p2)
            v_p_1_W=tf.expand_dims(tf.reshape(v_p_1_W,[opts["batch"],opts["p_len"],-1]),axis=1)
            v_p_2_W=tf.expand_dims(tf.reshape(v_p_2_W,[opts["batch"],opts["p_len"],-1]),axis=2)
            tanh=tf.tanh(v_p_1_W+v_p_2_W) # (b,p,p,h) 维度为1的地方自动广播
            print("tanh: {}".format(tanh))

            sj=tf.matmul(tf.reshape(tanh,[-1,opts["hidden_size"]]),tf.reshape(self.V_sm,[-1,1])) # (b*p*p,h)*(h,1)->(b*p*p,1)
            sj=tf.reshape(sj,[opts["batch"],opts["p_len"],opts["p_len"]])
            print("sj: {}".format(sj))
            ai=tf.nn.softmax(sj,axis=2) # (b,p,p)

            c=tf.matmul(ai,v_p) # (b,p,p)* (b,p,h) -> (b,p,h)

            # gate
            vp_c=tf.concat([v_p,c],axis=2) # (b,p,2h)
            sm_gate=self.mat_weight_mul(vp_c,self.W_sm_gate) # -> (b,p,2h)
            sm_gate=tf.sigmoid(sm_gate)
            vp_c_star=tf.multiply(sm_gate,vp_c) # -> (b,p,2h)
            print("vp_c_star: {}".format(vp_c_star))

        vp_c_star_unstack=tf.unstack(vp_c_star,axis=1)
        with tf.variable_scope("Self_match"):
            sm_fw_cell = self.DropoutWrappedLSTMCell(hidden_size=opts["hidden_size"], in_keep_prob=opts["dropout"])
            sm_bw_cell = self.DropoutWrappedLSTMCell(hidden_size=opts["hidden_size"], in_keep_prob=opts["dropout"])
            sm_output, _ , _ = tf.contrib.rnn.static_bidirectional_rnn(
                sm_fw_cell, sm_bw_cell, vp_c_star_unstack, dtype=tf.float32
            )
            h_P = tf.stack(sm_output, axis=1)  # (b,p,2h) fw+bw
        h_P=tf.nn.dropout(h_P,opts["dropout"])
        print("h_p: {}".format(h_P))
        with tf.variable_scope("Prediction_Layer"):
            print("Layer 4: Prediction Layer")

            # 公式：11a-11c
            tanh = tf.tanh(self.mat_weight_mul(u_q, self.W_q))  # (b,q,h)
            sj = self.mat_weight_mul(tanh, tf.reshape(self.V_q, [-1, 1]))  # (b,q,1)
            print("sjt: {}".format(sj))
            sj_trans = tf.transpose(sj, perm=[0, 2, 1])  # (b,1,q)为使用batch_dot 计算attention需要先转置
            ai = tf.nn.softmax(sj_trans, axis=2)
            r_q = tf.matmul(ai, u_q)  # (b,1,q)*(b,q,2h) => (b,1,2h)
            print("r_q:", r_q)
            # 公式：12a-12c
            tanh = tf.tanh(self.mat_weight_mul(h_P, self.W_p1) +
                           self.mat_weight_mul(r_q, self.W_p2))  # (b,p,2h)*(2h,h) => (b,p,h)
            sj = self.mat_weight_mul(tanh, tf.reshape(self.V_p, [-1, 1]))  # (b,p,1)
            sj_trans = tf.transpose(sj, perm=[0, 2, 1])  # (b,1,p)矩阵计算机attention 使用batch_dot 需要先转置
            ai = tf.nn.softmax(sj_trans, axis=2)
            r_p = tf.matmul(ai, h_P)  # (b,1,p) batch* (b,p,2h)

            # output
            encoder_out = tf.nn.relu(self.mat_weight_mul(r_p, self.W_predict))  # (b,p,2h)*(2h,h)=>(b,1,h)
            encoder_out = tf.nn.dropout(encoder_out, opts["dropout"])
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
            u_a, _, _ = tf.contrib.rnn.stack_bidirectional_rnn(a_fw_cells, a_bw_cells, a_emb_rs_us, dtype=tf.float32,
                                                                 scope="a_encoding")  # a_cell 要短些（后续计算需要）
            u_a = tf.nn.dropout(tf.stack(u_a, axis=1), keep_prob=opts["dropout"])  # (3b,a,h)
            print("u_a: {}".format(u_a))
            a_sj = self.mat_weight_mul(u_a,tf.reshape(self.V_a,[-1,1])) # (3b,a,h)*(h,1)=> (3b,a,1)
            a_ai = tf.nn.softmax(a_sj,axis=1) # (3b,a,1)
            a_ai= tf.transpose(a_ai,perm=[0,2,1]) # (3b,1,a) attention 转置
            a_output=tf.matmul(a_ai,u_a) # (3b,1,a)*(3b,a,h) => (3b,1,h)
            a_output=tf.squeeze(a_output) # (3b,h)
            a_output=tf.reshape(a_output,[opts["batch"],3,-1]) # (b,3,h)
            print("a_output: {}".format(a_output))

            # 结合encoder_out与answer
            score = tf.matmul(a_output, tf.transpose(encoder_out, perm=[0, 2, 1]))  # (b,3,h) batch* (b,h,1) => (b,3,1)
            score = tf.nn.softmax(tf.squeeze(score), axis=1, name="score")  # (b,3)
            print("socre: {}".format(score))

        print("complying...")
        print("loss...")
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
            labels=tf.constant([[1, 0, 0] for _ in range(opts["batch"])]), logits=score))
        print(loss)
        print("optimizer...")
        optimizer = tf.train.AdamOptimizer(name="opt").minimize(loss)
        print("opt: {}".format(optimizer))
        print("predict..")
        predict = tf.argmax(score, axis=1, name="pred")  # (10,1)
        print("pred: {}".format(predict))
        print("dict...")

        test_q = tf.squeeze(query[:, 0])
        test_p = tf.squeeze(para[:, 0])
        test_a = tf.squeeze(ans[:, 0, 0])
        test_op = [test_p, test_a, test_q]
        print("test_op(用于快速测试feed_dict): {}".format(test_op))

        tensor_dict = {
            "q": query,
            "p": para,
            "a": ans
        }
        return loss, optimizer, predict, tensor_dict, test_op


