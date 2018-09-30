import json
import math

import numpy as np
import  tensorflow as tf
import tensorflow.contrib as tc



class BiDAF():
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
        self.opts = json.load(open("model/dureader_config.json"))
        opts = self.opts

        # self.a_attention = nn.Linear(embedding_size, 1, bias=False)
        stdv = 1. / math.sqrt(opts["embedding_size"])
        self.a_attention = tf.Variable(tf.random_uniform(minval=-stdv, maxval=stdv, shape=[opts["hidden_size"], 1]))

        """
        prediction layer
       """
        # self.Wq = nn.Linear(2 * encoder_size, encoder_size, bias=False)
        self.Wq = tf.Variable(
            tf.random_uniform(minval=-stdv, maxval=stdv, shape=[2 * opts["hidden_size"], opts["hidden_size"]]))
        # self.vq = nn.Linear(encoder_size, 1, bias=False)
        self.vq = tf.Variable(tf.random_uniform(minval=-stdv, maxval=stdv, shape=[opts["hidden_size"], 1]))
        # self.Wp1 = nn.Linear(2 * encoder_size, encoder_size, bias=False)
        self.Wp1 = tf.Variable(
            tf.random_uniform(minval=-stdv, maxval=stdv, shape=[2 * opts["hidden_size"], opts["hidden_size"]]))
        # self.Wp2 = nn.Linear(2 * encoder_size, encoder_size, bias=False)
        self.Wp2 = tf.Variable(
            tf.random_uniform(minval=-stdv, maxval=stdv, shape=[2 * opts["hidden_size"], opts["hidden_size"]]))
        # self.vp = nn.Linear(encoder_size, 1, bias=False)
        self.vp = tf.Variable(tf.random_uniform(minval=-stdv, maxval=stdv, shape=[opts["hidden_size"], 1]))
        # self.prediction = nn.Linear(2 * encoder_size, embedding_size, bias=False)
        self.prediction = tf.Variable(
            tf.random_uniform(minval=-stdv, maxval=stdv, shape=[2 * opts["hidden_size"], opts["hidden_size"]]))



    def build(self):
        print("building model...")
        opts = self.opts

        # placeholder
        # FIXME: batch should be changed to None
        query = tf.placeholder(dtype=tf.int32, shape=[opts["batch"], opts["q_len"]], name="query")
        passage = tf.placeholder(dtype=tf.int32, shape=[opts["batch"], opts["p_len"]], name="para")
        answer = tf.placeholder(dtype=tf.int32, shape=[opts["batch"], 3, opts["alt_len"]], name="ans")


        with tf.variable_scope('word_embedding'):
            self.emb = tf.get_variable(
                'word_embeddings',
                shape=(opts["vocab_size"], opts["embedding_size"]),
                initializer=tf.constant_initializer(np.random.rand(opts["vocab_size"], opts["embedding_size"])
            ))
            q_embedding = tf.nn.embedding_lookup(params=self.emb, ids=query)
            p_embedding = tf.nn.embedding_lookup(params=self.emb, ids=passage)
            a_embeddings = tf.nn.embedding_lookup(params=self.emb, ids=answer)

        print("layer1: encoding layer")
        with tf.variable_scope("a_encoding"):
            print("a encoding...")
            cell_fw_a = tc.rnn.MultiRNNCell([tc.rnn.LSTMCell(num_units=opts["hidden_size"]/2) for _ in range(1)])
            cell_bw_a = tc.rnn.MultiRNNCell([tc.rnn.LSTMCell(num_units=opts["hidden_size"]/2) for _ in range(1)])
            a_embeddings_r = tf.reshape(a_embeddings, shape=[-1, opts["alt_len"], opts["embedding_size"]])  # (3b,a,d)
            a_encoder, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw_a, cell_bw_a, a_embeddings_r, dtype=tf.float32)
            a_encoder = tf.concat(a_encoder, axis=2)  # (3b,a,h)
            print("a_encoder: {}".format(a_encoder))
            a_score = tf.nn.softmax(self.weight_matmul(a_encoder, self.a_attention), axis=1)  # (3b,a,1)
            a_output = tf.matmul(tf.transpose(a_score, perm=[0, 2, 1]), a_encoder)  # (3b,1,a) b* (3b,a,h) -> (3b,1,h)
            a_output = tf.squeeze(a_output)  # (3b,h)
            a_embedding = tf.reshape(a_output, shape=[-1, 3, opts["hidden_size"]])  # (b,3,h)
            print("a_embedding: {}".format(a_embedding))

        with tf.variable_scope("qp_encoding"):
            print("pq encoding...")
            with tf.variable_scope('passage_encoding'):
                f_cell=tc.rnn.MultiRNNCell([tc.rnn.LSTMCell(num_units=opts["hidden_size"], state_is_tuple=True) for _ in range(1)])
                b_cell=tc.rnn.MultiRNNCell([tc.rnn.LSTMCell(num_units=opts["hidden_size"], state_is_tuple=True) for _ in range(1)])
                sep_p_encodes, _ = tf.nn.bidirectional_dynamic_rnn(f_cell,b_cell,p_embedding,dtype=tf.float32)
                sep_p_encodes=tf.concat(sep_p_encodes,2)
            with tf.variable_scope('question_encoding'):
                f_cell = tc.rnn.MultiRNNCell([tc.rnn.LSTMCell(num_units=opts["hidden_size"], state_is_tuple=True) for _ in range(1)])
                b_cell = tc.rnn.MultiRNNCell([tc.rnn.LSTMCell(num_units=opts["hidden_size"], state_is_tuple=True) for _ in range(1)])
                sep_q_encodes, _ = tf.nn.bidirectional_dynamic_rnn(f_cell, b_cell, q_embedding, dtype=tf.float32)
                sep_q_encodes=tf.concat(sep_q_encodes,2)
            print("q_encodes: {}".format(sep_q_encodes))
            print("p_encodes: {}".format(sep_p_encodes))


        with tf.variable_scope("matching"):
            print("matching layer:")
            with tf.variable_scope("bidaf"):
                print("bidaf")
                sim_matrix=tf.matmul(sep_p_encodes,sep_q_encodes,transpose_b=True) #b在进行乘法计算前对后两维进行转置
                print("sim_matrix: {}".format(sim_matrix)) # (b,p,h) (b,h,q)-> (b,p,q)

                # 计算context2question_attn
                c2q_sim_matrix=tf.nn.softmax(sim_matrix,2) # -1 表示最后一维(此处为q)
                context2qusetion_attn=tf.matmul(c2q_sim_matrix,sep_q_encodes) # (b,p,q) (b,q,2h)->(b,p,2h)

                # 计算question2context_attn
                maxP_sim_mat=tf.reduce_max(sim_matrix,axis=2) # (b,p)
                maxP_sim_mat=tf.expand_dims(maxP_sim_mat,axis=1) # (b,1,p)
                b=tf.nn.softmax(maxP_sim_mat,2) # (b,1,p)
                question2context_attn=tf.matmul(b,sep_p_encodes) # (b,1,p) (b,p,2h) -> (b,1,2h)
                question2context_attn=tf.tile(question2context_attn,[1,opts["p_len"],1]) # (b,p,2h) 在维度1上复制

                match_p_encodes=tf.concat([sep_p_encodes, # (b,p,2h)
                                       context2qusetion_attn, # (b,p,2h)
                                       sep_p_encodes*context2qusetion_attn, # 对应位置相乘
                                       sep_p_encodes*question2context_attn],axis=2)
                print("match_layer: {}".format(match_p_encodes)) # (b,p,4h)
                
                
            with tf.variable_scope("fusion"):
                print("fusion:")
                f_cell=tc.rnn.MultiRNNCell([tc.rnn.LSTMCell(num_units=opts["hidden_size"]) for _ in range(1)])
                b_cell=tc.rnn.MultiRNNCell([tc.rnn.LSTMCell(num_units=opts["hidden_size"]) for _ in range(1)])
                fuse_p_encodes,_ =tf.nn.bidirectional_dynamic_rnn(f_cell,b_cell,match_p_encodes,dtype=tf.float32)
                fuse_p_encodes=tf.concat(fuse_p_encodes,axis=2)

        with tf.variable_scope("prediction_layer"):
            print("prediction layer:")
            print("sep_q_encodes: {}".format(sep_q_encodes)) # (b,q,2h)
            print("fuse_p_encodes: {}".format(fuse_p_encodes)) # (b,p,2h)

            tanh = tf.nn.tanh(self.weight_matmul(sep_q_encodes, self.Wq))
            sj = tf.transpose(self.weight_matmul(tanh, self.vq), perm=[0, 2, 1])

            ai = tf.nn.softmax(sj, 2)  # (b,1,q)
            rq = tf.matmul(ai, sep_q_encodes)  # (b,1,q) (b,q,2h) -> (b,1,2h)


            add = self.weight_matmul(fuse_p_encodes, self.Wp1) + self.weight_matmul(rq, self.Wp2)
            sj = tf.transpose(self.weight_matmul(add, self.vp), perm=[0, 2, 1])
            sj = tf.nn.softmax(sj, 2)  # (b,1,p)
            rp = tf.matmul(sj, fuse_p_encodes)  # (b,1,p) (b,p,2h) -> (b,1,2h)
            print("rp: {}".format(rp))

            full = self.weight_matmul(rp, self.prediction)  # (b,1,2h) (2h,h) -> (b,1,h)
            encoder_output = tf.nn.dropout(tf.nn.leaky_relu(full), opts["dropout"])  # (b,1,h)
            bmm = tf.matmul(a_embedding, tf.transpose(encoder_output, [0, 2, 1]))  # (b,3,h) (b,h,1) -> (b,3,1)
            bmm = tf.squeeze(bmm)  # (b,3)
            score = tf.nn.softmax(bmm, axis=1)
            print("socre: {}".format(score))

        print("complying...")
        print("loss...")
        loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf.constant([[1, 0, 0] for _ in range(opts["batch"])]),
                                                       logits=score))
        print("optimizer...")
        optimizer = tf.train.AdamOptimizer(learning_rate=opts["learning_rate"],name="opt").minimize(loss)
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






