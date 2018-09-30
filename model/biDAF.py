import json
import  tensorflow as tf

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
        self.opts = json.load(open("model/config.json"))
        opts = self.opts

    def build(self):
        print("building model...")
        opts = self.opts

        # placeholder
        # FIXME: batch should be changed to None
        query = tf.placeholder(dtype=tf.int32, shape=[opts["batch"], opts["q_len"]], name="query")
        passage = tf.placeholder(dtype=tf.int32, shape=[opts["batch"], opts["p_len"]], name="para")
        answer = tf.placeholder(dtype=tf.int32, shape=[opts["batch"], 3, opts["alt_len"]], name="ans")

        q_embedding = tf.nn.embedding_lookup(params=self.emb, ids=query)
        p_embedding = tf.nn.embedding_lookup(params=self.emb, ids=passage)
        a_embeddings = tf.nn.embedding_lookup(params=self.emb, ids=answer)

        print("layer1: encoding layer")
        with tf.variable_scope("a_encoding"):
            print("a encoding...")
            cell_fw_a = tf.nn.rnn_cell.GRUCell(num_units=opts["hidden_size"] / 2)
            cell_bw_a = tf.nn.rnn_cell.GRUCell(num_units=opts["hidden_size"] / 2)
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
            print("pp encoding...")
            cell_fw = tf.nn.rnn_cell.GRUCell(num_units=opts["hidden_size"])
            cell_bw = tf.nn.rnn_cell.GRUCell(num_units=opts["hidden_size"])

            # hq, _ = self.q_encoder(p_embedding)
            # hq = F.dropout(hq, self.drop_out)
            hq, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, q_embedding, dtype=tf.float32)  # (b,q,2h)
            hq = tf.concat(hq, axis=2)
            hq = tf.nn.dropout(hq, keep_prob=opts["dropout"])
            # hp, _ = self.p_encoder(q_embedding)
            # hp = F.dropout(hp, self.drop_out)
            hp, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, p_embedding, dtype=tf.float32)  # (b,p,2h)
            hp = tf.concat(hp, axis=2)
            hp = tf.nn.dropout(hp, keep_prob=opts["dropout"])
            print("p/q_encoder: {}".format(hq))
