
import time
import numpy as np
# import tensorflow as tf
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

import config
from utils import os_utils
from tf_common.optimizer import *
from tf_common.nn_module import word_dropout, mlp_layer
from tf_common.nn_module import encode, attend


def sigmoid(x):
    return 1./(1.+np.exp(-x))


class BaseModel(object):
    def __init__(self, params, logger, init_embedding_matrix=None):
        self.params = params # 参数列表
        self.logger = logger
        self.init_embedding_matrix = init_embedding_matrix
        self.model_name = self.params["model_name"]
        self.threshold = self.params["threshold"]
        self.calibration_model = None
        # os_utils._makedirs(self.params["offline_model_dir"], force=True)

        self._init_tf_vars()
        # 该方法父类未实现, pass
        # 以dssm为例，matching_feature返回两类网络的query和doc的相似度值：
        # 一个是char网络，一个是word网络，其中每个网络的相似度值均为[cos_sim, dot_product, eculiden_distance]
        self.matching_features_word, self.matching_features_char = self._get_matching_features()
        # logits默认由fc_hidden_units指定层数的的FC层和一层无激活，单神经元的FC生成，proba由logits使用sigmoid激活生成
        self.logits, self.proba = self._get_prediction()
        # _get_loss(）默认使用基本的sigmoid交叉熵函数，且根据参数决定是否加入L2正则
        self.loss = self._get_loss()
        # 根据配置不同，生成各种optimizer
        self.train_op = self._get_train_op()
        # 记录logloss，augmentDropout，learningRate等信息
        self.summary = self._get_summary()

        # 至此，model的网络构建完成，且变量初始化完成，由main调用fit函数，塞入经过分batch后的数据，生成feed_dict开始训练
        self.sess, self.saver = self._init_session()
        self.train_writer = tf.summary.FileWriter(self.params["summary_dir"] + '/train', self.sess.graph)
        self.test_writer = tf.summary.FileWriter(self.params["summary_dir"] + '/test')
        
        
    def _init_tf_vars(self):
        #### training flag
        self.training = tf.placeholder(tf.bool, shape=[], name="training")
        #### labels
        self.labels = tf.placeholder(tf.float32, shape=[None], name="labels")
        #### word
        self.seq_word_left = tf.placeholder(tf.int32, shape=[None, None], name="seq_word_left")
        self.seq_word_right = tf.placeholder(tf.int32, shape=[None, None], name="seq_word_right")
        #### char
        self.seq_char_left = tf.placeholder(tf.int32, shape=[None, None], name="seq_char_left")
        self.seq_char_right = tf.placeholder(tf.int32, shape=[None, None], name="seq_char_right")
        #### word len
        self.seq_len_word_left = tf.placeholder(tf.int32, shape=[None], name="seq_len_word_left")
        self.seq_len_word_right = tf.placeholder(tf.int32, shape=[None], name="seq_len_word_right")
        #### char len
        self.seq_len_char_left = tf.placeholder(tf.int32, shape=[None], name="seq_len_char_left")
        self.seq_len_char_right = tf.placeholder(tf.int32, shape=[None], name="seq_len_char_right")

        #### features
        self.features = tf.placeholder(tf.float32, shape=[None, self.params["num_features"]], name="features")

        #### training
        self.global_step = tf.Variable(0, trainable=False)
        self.learning_rate = tf.train.exponential_decay(self.params["init_lr"], self.global_step,
                                                        self.params["decay_steps"], self.params["decay_rate"])
        self.augmentation_dropout = tf.train.exponential_decay(self.params["augmentation_init_dropout"], self.global_step,
                                                               self.params["augmentation_dropout_decay_steps"],
                                                               self.params["augmentation_dropout_decay_rate"])
        self.augmentation_permutation = tf.train.exponential_decay(self.params["augmentation_init_permutation"],
                                                               self.global_step,
                                                               self.params["augmentation_permutation_decay_steps"],
                                                               self.params["augmentation_permutation_decay_rate"])


    # 父类提供的默认embedding_matrix, 当init_embedding_matrix没有提供用于初始化embedding矩阵的分布的参数时，
    # 使用一些默认参数初始化
    def _get_embedding_matrix(self, granularity="word"):
        if self.init_embedding_matrix[granularity] is None:
            std = 0.1
            minval = -std
            maxval = std
            emb_matrix = tf.Variable(
                tf.random_uniform(
                    [self.params["max_num_%s" % granularity] + 1, self.params["embedding_dim_%s" % granularity]],
                    minval, maxval,
                    seed=self.params["random_seed"],
                    dtype=tf.float32))
        else:
            emb_matrix = tf.Variable(self.init_embedding_matrix[granularity],
                                     trainable=self.params["embedding_trainable"])
        return emb_matrix

    # 语义层。emed + dropout -> encode -> attend -> FC（relu)
    # 其中encode部分，根据给定的method不同，分别使用fast text，project，text_cnn, text_rnn来进行编码
    def _semantic_feature_layer(self, seq_input, seq_len, granularity="word", reuse=False):
        # granularity应该指示的是模型使用基于单词的model还是基于character的model
        assert granularity in ["char", "word"]
        #### embed
        emb_matrix = self._get_embedding_matrix(granularity)
        emb_seq = tf.nn.embedding_lookup(emb_matrix, seq_input)

        #### dropout
        random_seed = np.random.randint(10000000)
        emb_seq = word_dropout(emb_seq,
                               training=self.training,
                               dropout=self.params["embedding_dropout"],
                               seed=random_seed)

        #### encode
        input_dim = self.params["embedding_dim"]
        enc_seq = encode(emb_seq, method=self.params["encode_method"],
                         input_dim=input_dim,
                         params=self.params,
                         sequence_length=seq_len,
                         mask_zero=self.params["embedding_mask_zero"],
                         scope_name=self.model_name + "enc_seq_%s"%granularity, reuse=reuse,
                         training=self.training)

        #### attend
        feature_dim = self.params["encode_dim"]
        context = None
        att_seq = attend(enc_seq, context=context,
                         encode_dim=self.params["encode_dim"],
                         feature_dim=feature_dim,
                         attention_dim=self.params["attention_dim"],
                         method=self.params["attend_method"],
                         scope_name=self.model_name + "att_seq_%s"%granularity,
                         reuse=reuse, num_heads=self.params["attention_num_heads"])

        #### MLP nonlinear projection
        sem_seq = mlp_layer(att_seq, fc_type=self.params["fc_type"],
                            hidden_units=self.params["fc_hidden_units"],
                            dropouts=self.params["fc_dropouts"],
                            scope_name=self.model_name + "sem_seq_%s"%granularity,
                            reuse=reuse,
                            training=self.training,
                            seed=self.params["random_seed"])

        return emb_seq, enc_seq, att_seq, sem_seq


    def _interaction_semantic_feature_layer(self, seq_input_left, seq_input_right, seq_len_left, seq_len_right, granularity="word"):
        assert granularity in ["char", "word"]
        #### embed
        emb_matrix = self._get_embedding_matrix(granularity)
        emb_seq_left = tf.nn.embedding_lookup(emb_matrix, seq_input_left)
        emb_seq_right = tf.nn.embedding_lookup(emb_matrix, seq_input_right)

        #### dropout
        random_seed = np.random.randint(10000000)
        emb_seq_left = word_dropout(emb_seq_left,
                               training=self.training,
                               dropout=self.params["embedding_dropout"],
                               seed=random_seed)
        random_seed = np.random.randint(10000000)
        emb_seq_right = word_dropout(emb_seq_right,
                                    training=self.training,
                                    dropout=self.params["embedding_dropout"],
                                    seed=random_seed)

        #### encode
        input_dim = self.params["embedding_dim"]
        enc_seq_left = encode(emb_seq_left, method=self.params["encode_method"],
                              input_dim=input_dim,
                              params=self.params,
                              sequence_length=seq_len_left,
                              mask_zero=self.params["embedding_mask_zero"],
                              scope_name=self.model_name + "enc_seq_%s"%granularity, reuse=False,
                              training=self.training)
        enc_seq_right = encode(emb_seq_right, method=self.params["encode_method"],
                               input_dim=input_dim,
                               params=self.params,
                               sequence_length=seq_len_right,
                               mask_zero=self.params["embedding_mask_zero"],
                               scope_name=self.model_name + "enc_seq_%s" % granularity, reuse=True,
                               training=self.training)

        #### attend
        # [batchsize, s1, s2]
        att_mat = tf.einsum("abd,acd->abc", enc_seq_left, enc_seq_right)
        feature_dim = self.params["encode_dim"] + self.params["max_seq_len_%s"%granularity]
        att_seq_left = attend(enc_seq_left, context=att_mat, feature_dim=feature_dim,
                                   method=self.params["attend_method"],
                                   scope_name=self.model_name + "att_seq_%s"%granularity,
                                   reuse=False)
        att_seq_right = attend(enc_seq_right, context=tf.transpose(att_mat), feature_dim=feature_dim,
                              method=self.params["attend_method"],
                              scope_name=self.model_name + "att_seq_%s" % granularity,
                              reuse=True)

        #### MLP nonlinear projection
        sem_seq_left = mlp_layer(att_seq_left, fc_type=self.params["fc_type"],
                                 hidden_units=self.params["fc_hidden_units"],
                                 dropouts=self.params["fc_dropouts"],
                                 scope_name=self.model_name + "sem_seq_%s"%granularity,
                                 reuse=False,
                                 training=self.training,
                                 seed=self.params["random_seed"])
        sem_seq_right = mlp_layer(att_seq_right, fc_type=self.params["fc_type"],
                                  hidden_units=self.params["fc_hidden_units"],
                                  dropouts=self.params["fc_dropouts"],
                                  scope_name=self.model_name + "sem_seq_%s" % granularity,
                                  reuse=True,
                                  training=self.training,
                                  seed=self.params["random_seed"])

        return emb_seq_left, enc_seq_left, att_seq_left, sem_seq_left, \
                emb_seq_right, enc_seq_right, att_seq_right, sem_seq_right


    def _get_matching_features(self):
        pass


    """
    预测部分由 FC(relu) concat fearture_word或feature_char + dropout -> FC(relu) -> FC(无激活函数） 
    """
    def _get_prediction(self):
        with tf.name_scope(self.model_name + "/"):
            with tf.name_scope("prediction"):
                lst = []
                # granularity 指示了使用基于词的网络还是基于字符的网络。从main函数的参数来看，这个应该只能指定一个
                if "word" in self.params["granularity"]:
                    lst.append(self.matching_features_word)
                if "char" in self.params["granularity"]:
                    lst.append(self.matching_features_char)
                    # main函数中的配置默认为False, dssm没有对该参数进行更新，默认不use_features
                if self.params["use_features"]:
                    # mlp_layer是多层感知机, 包含fc，dense和resent三种类型的网络层单元
                    out_0 = mlp_layer(self.features, fc_type=self.params["fc_type"],
                                      hidden_units=self.params["fc_hidden_units"],
                                      dropouts=self.params["fc_dropouts"],
                                      scope_name=self.model_name + "mlp_features",
                                      reuse=False,
                                      training=self.training,
                                      seed=self.params["random_seed"])
                    lst.append(out_0)
                out = tf.concat(lst, axis=-1)
                out = tf.layers.Dropout(self.params["final_dropout"])(out, training=self.training)
                # 这里为何又有fc_hidden_units个层的全连接网络？？
                out = mlp_layer(out, fc_type=self.params["fc_type"],
                                hidden_units=self.params["fc_hidden_units"],
                                dropouts=self.params["fc_dropouts"],
                                scope_name=self.model_name + "mlp",
                                reuse=False,
                                training=self.training,
                                seed=self.params["random_seed"])
                # prediction这最后的单神经元，不含集货函数的层是干嘛的？
                logits = tf.layers.dense(out, 1, activation=None,
                                         kernel_initializer=tf.glorot_uniform_initializer(
                                         seed=self.params["random_seed"]),
                                         name=self.model_name + "logits")
                logits = tf.squeeze(logits, axis=1)
                proba = tf.nn.sigmoid(logits)

        return logits, proba


    def _get_loss(self):
        with tf.name_scope(self.model_name + "/"):
            with tf.name_scope("loss"):
                loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.labels, logits=self.logits)
                loss = tf.reduce_mean(loss, name="log_loss")
                if self.params["l2_lambda"] > 0:
                    l2_losses = tf.add_n(
                        [tf.nn.l2_loss(v) for v in tf.trainable_variables() if "bias" not in v.name]) * self.params[
                                    "l2_lambda"]
                    loss = loss + l2_losses
        return loss


    def _get_train_op(self):
        with tf.name_scope(self.model_name + "/"):
            with tf.name_scope("optimization"):
                if self.params["optimizer_type"] == "lazynadam":
                    optimizer = LazyNadamOptimizer(learning_rate=self.learning_rate, beta1=self.params["beta1"],
                                                   beta2=self.params["beta2"], epsilon=1e-8,
                                                   schedule_decay=self.params["schedule_decay"])
                elif self.params["optimizer_type"] == "adam":
                    optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate,
                                                       beta1=self.params["beta1"],
                                                       beta2=self.params["beta2"], epsilon=1e-8)
                elif self.params["optimizer_type"] == "lazyadam":
                    optimizer = tf.contrib.opt.LazyAdamOptimizer(learning_rate=self.learning_rate,
                                                                 beta1=self.params["beta1"],
                                                                 beta2=self.params["beta2"], epsilon=1e-8)
                elif self.params["optimizer_type"] == "adagrad":
                    optimizer = tf.train.AdagradOptimizer(learning_rate=self.learning_rate,
                                                          initial_accumulator_value=1e-7)
                elif self.params["optimizer_type"] == "adadelta":
                    optimizer = tf.train.AdadeltaOptimizer(learning_rate=self.learning_rate)
                elif self.params["optimizer_type"] == "gd":
                    optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate)
                elif self.params["optimizer_type"] == "momentum":
                    optimizer = tf.train.MomentumOptimizer(learning_rate=self.learning_rate, momentum=0.95)
                elif self.params["optimizer_type"] == "rmsprop":
                    optimizer = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate, decay=0.9,
                                                          momentum=0.9, epsilon=1e-8)
                elif self.params["optimizer_type"] == "lazypowersign":
                    optimizer = LazyPowerSignOptimizer(learning_rate=self.learning_rate)
                elif self.params["optimizer_type"] == "lazyaddsign":
                    optimizer = LazyAddSignOptimizer(learning_rate=self.learning_rate)
                elif self.params["optimizer_type"] == "lazyamsgrad":
                    optimizer = LazyAMSGradOptimizer(learning_rate=self.learning_rate, beta1=self.params["beta1"],
                                                     beta2=self.params["beta2"], epsilon=1e-8)

                # todo get_collection和GrahKeys是什么意思？
                update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                with tf.control_dependencies(update_ops):
                    train_op = optimizer.minimize(self.loss, global_step=self.global_step)
        return train_op


    def _get_summary(self):
        with tf.name_scope(self.model_name + "/"):
            # todo 什么是augmentation_dropout?
            tf.summary.scalar("augmentation_dropout", self.augmentation_dropout)
            tf.summary.scalar("logloss", self.loss)
            tf.summary.scalar("lr", self.learning_rate)
            # error: https://blog.csdn.net/u012436149/article/details/53894364
            # summary = tf.summary.merge_all()
            summary = tf.summary.merge(
                tf.get_collection(tf.GraphKeys.SUMMARIES, self.model_name)
            )
        return summary


    def _init_session(self):
        # todo 看下这几个参数的意义
        config = tf.ConfigProto(device_count={"gpu": 1})
        config.gpu_options.allow_growth = True
        config.intra_op_parallelism_threads = 4
        config.inter_op_parallelism_threads = 4
        sess = tf.Session(config=config)
        sess.run(tf.global_variables_initializer())
        # max_to_keep=None, keep all the models
        # todo train.saver的意义
        saver = tf.train.Saver(max_to_keep=None)
        return sess, saver


    def save_session(self):
        self.saver.save(self.sess, self.params["offline_model_dir"] + "/model.checkpoint")


    def restore_session(self):
        self.saver.restore(self.sess, self.params["offline_model_dir"] + "/model.checkpoint")


    def _get_batch_index(self, seq, step):
        n = len(seq)
        res = []
        for i in range(0, n, step):
            res.append(seq[i:i + step])
        # last batch
        if len(res) * step < n:
            res.append(seq[len(res) * step:])
        return res


    def _get_pos_neg_ind(self, label):
        length = len(label)
        pos_ind_tmp = np.where(label == 1)[0]
        inds = np.zeros((len(pos_ind_tmp) * length, 2), dtype=int)
        inds[:, 0] = np.tile(pos_ind_tmp, length)
        inds[:, 1] = list(range(length)) * len(pos_ind_tmp)
        mask = inds[:, 0] != inds[:, 1]
        pos_ind = inds[mask, 0]
        neg_ind = inds[mask, 1]
        return pos_ind, neg_ind


    def _get_feed_dict(self, X, idx, Q, construct_neg=False, training=False, symmetric=False):
        if training:
            if construct_neg:
                q1 = X["q1"][idx]
                q2 = X["q2"][idx]
                # for label=1 sample, construct negative sample within batch
                pos_ind, neg_ind = self._get_pos_neg_ind(X["label"][idx])
                # original & symmetric
                feed_dict = {
                    self.seq_word_left: np.vstack([Q["words"][q1],
                                                   Q["words"][X["q1"][idx[pos_ind]]],
                                                   Q["words"][X["q1"][idx[neg_ind]]],
                                                   Q["words"][q2],
                                                   Q["words"][X["q2"][idx[neg_ind]]],
                                                   Q["words"][X["q2"][idx[pos_ind]]]
                                                   ]),
                    self.seq_word_right: np.vstack([Q["words"][q2],
                                                    Q["words"][X["q2"][idx[neg_ind]]],
                                                    Q["words"][X["q2"][idx[pos_ind]]],
                                                    Q["words"][q1],
                                                    Q["words"][X["q1"][idx[pos_ind]]],
                                                    Q["words"][X["q1"][idx[neg_ind]]],
                                                    ]),
                    self.seq_char_left: np.vstack([Q["chars"][q1],
                                                   Q["chars"][X["q1"][idx[pos_ind]]],
                                                   Q["chars"][X["q1"][idx[neg_ind]]],
                                                   Q["chars"][q2],
                                                   Q["chars"][X["q2"][idx[neg_ind]]],
                                                   Q["chars"][X["q2"][idx[pos_ind]]]
                                                   ]),
                    self.seq_char_right: np.vstack([Q["chars"][q2],
                                                    Q["chars"][X["q2"][idx[neg_ind]]],
                                                    Q["chars"][X["q2"][idx[pos_ind]]],
                                                    Q["chars"][q1],
                                                    Q["chars"][X["q1"][idx[pos_ind]]],
                                                    Q["chars"][X["q1"][idx[neg_ind]]]
                                                    ]),
                    self.labels: np.hstack([X["label"][idx],
                                            np.zeros(len(pos_ind)),
                                            np.zeros(len(pos_ind)),
                                            X["label"][idx],
                                            np.zeros(len(pos_ind)),
                                            np.zeros(len(pos_ind))
                                            ]),
                    self.training: training,
                }
            else:
                q1 = X["q1"][idx]
                q2 = X["q2"][idx]
                feed_dict = {
                    self.seq_word_left: np.vstack([Q["words"][q1],
                                                   Q["words"][q2],
                                                   ]),
                    self.seq_word_right: np.vstack([Q["words"][q2],
                                                    Q["words"][q1],
                                                    ]),
                    self.seq_char_left: np.vstack([Q["chars"][q1],
                                                   Q["chars"][q2],
                                                   ]),
                    self.seq_char_right: np.vstack([Q["chars"][q2],
                                                    Q["chars"][q1],
                                                    ]),
                    self.seq_len_word_left: np.hstack([Q["seq_len_word"][q1],
                                                       Q["seq_len_word"][q2],
                                                       ]),
                    self.seq_len_word_right: np.hstack([Q["seq_len_word"][q2],
                                                        Q["seq_len_word"][q1],
                                                        ]),
                    self.seq_len_char_left: np.hstack([Q["seq_len_char"][q1],
                                                       Q["seq_len_char"][q2],
                                                       ]),
                    self.seq_len_char_right: np.hstack([Q["seq_len_char"][q2],
                                                        Q["seq_len_char"][q1],
                                                        ]),
                    self.labels: np.hstack([X["label"][idx],
                                            X["label"][idx],
                                            ]),
                    self.training: training,
                }
                if self.params["use_features"]:
                    feed_dict.update({
                        self.features: np.vstack([X["features"][idx],
                                                  X["features"][idx],
                                                  ]),
                    })
        elif not symmetric:
            q1 = X["q1"][idx]
            q2 = X["q2"][idx]
            feed_dict = {
                self.seq_word_left: Q["words"][q1],
                self.seq_word_right: Q["words"][q2],
                self.seq_char_left: Q["chars"][q1],
                self.seq_char_right: Q["chars"][q2],
                self.seq_len_word_left: Q["seq_len_word"][q1],
                self.seq_len_word_right: Q["seq_len_word"][q2],
                self.seq_len_char_left: Q["seq_len_char"][q1],
                self.seq_len_char_right: Q["seq_len_char"][q2],
                self.labels: X["label"][idx],
                self.training: training,
            }
            if self.params["use_features"]:
                feed_dict.update({
                    self.features: X["features"][idx],
                })
        else:
            q1 = X["q1"][idx]
            q2 = X["q2"][idx]
            feed_dict = {
                self.seq_word_left: np.vstack([Q["words"][q1],
                                               Q["words"][q2],
                                               ]),
                self.seq_word_right: np.vstack([Q["words"][q2],
                                                Q["words"][q1],
                                                ]),
                self.seq_char_left: np.vstack([Q["chars"][q1],
                                               Q["chars"][q2],
                                               ]),
                self.seq_char_right: np.vstack([Q["chars"][q2],
                                                Q["chars"][q1],
                                                ]),
                self.seq_len_word_left: np.hstack([Q["seq_len_word"][q1],
                                                   Q["seq_len_word"][q2],
                                                   ]),
                self.seq_len_word_right: np.hstack([Q["seq_len_word"][q2],
                                                    Q["seq_len_word"][q1],
                                                    ]),
                self.seq_len_char_left: np.hstack([Q["seq_len_char"][q1],
                                                   Q["seq_len_char"][q2],
                                                   ]),
                self.seq_len_char_right: np.hstack([Q["seq_len_char"][q2],
                                                    Q["seq_len_char"][q1],
                                                    ]),
                self.labels: np.hstack([X["label"][idx],
                                        X["label"][idx],
                                        ]),
                self.training: training,
            }
            if self.params["use_features"]:
                feed_dict.update({
                    self.features: np.vstack([X["features"][idx],
                                              X["features"][idx],
                                              ]),
                })
        # augmentation
        if training:
            if self.params["augmentation_init_dropout"] > 0:
                self._dropout_augmentation(feed_dict)
            if self.params["augmentation_init_permutation"]:
                self._permutation_augmentation(feed_dict)

        return feed_dict


    def _dropout(self, val_arr, ind_arr, p, value):
        new_arr = np.array(val_arr)
        drop = np.empty(val_arr.shape, dtype=np.bool)
        for i in range(val_arr.shape[0]):
            drop[i, :ind_arr[i]] = np.random.choice([True, False], ind_arr[i], p=[p, 1 - p])
        new_arr[drop] = value
        return new_arr


    def _dropout_augmentation(self, feed_dict):
        p = self.sess.run(self.augmentation_dropout)
        if p <= self.params["augmentation_min_dropout"]:
            return

        dropout_data = self._dropout(val_arr=feed_dict[self.seq_word_left],
                                     ind_arr=feed_dict[self.seq_len_word_left],
                                     p=p, value=config.MISSING_INDEX_WORD)
        feed_dict[self.seq_word_left] = np.vstack([
            feed_dict[self.seq_word_left],
            dropout_data,
        ])

        dropout_data = self._dropout(val_arr=feed_dict[self.seq_word_right],
                                     ind_arr=feed_dict[self.seq_len_word_right],
                                     p=p, value=config.MISSING_INDEX_WORD)
        feed_dict[self.seq_word_right] = np.vstack([
            feed_dict[self.seq_word_right],
            dropout_data,
        ])

        dropout_data = self._dropout(val_arr=feed_dict[self.seq_char_left],
                                     ind_arr=feed_dict[self.seq_len_char_left],
                                     p=p, value=config.MISSING_INDEX_CHAR)
        feed_dict[self.seq_char_left] = np.vstack([
            feed_dict[self.seq_char_left],
            dropout_data,
        ])

        dropout_data = self._dropout(val_arr=feed_dict[self.seq_char_right],
                                     ind_arr=feed_dict[self.seq_len_char_right],
                                     p=p, value=config.MISSING_INDEX_CHAR)
        feed_dict[self.seq_char_right] = np.vstack([
            feed_dict[self.seq_char_right],
            dropout_data,
        ])

        # double others
        feed_dict[self.seq_len_word_left] = np.tile(feed_dict[self.seq_len_word_left], 2)
        feed_dict[self.seq_len_word_right] = np.tile(feed_dict[self.seq_len_word_right], 2)
        feed_dict[self.seq_len_char_left] = np.tile(feed_dict[self.seq_len_char_left], 2)
        feed_dict[self.seq_len_char_right] = np.tile(feed_dict[self.seq_len_char_right], 2)
        feed_dict[self.labels] = np.tile(feed_dict[self.labels], 2)
        if self.params["use_features"]:
            feed_dict[self.features] = np.tile(feed_dict[self.features], [2, 1])


    def _permutation(self, val_arr, ind_arr, p):
        if np.random.random() < p:
            new_arr = np.array(val_arr)
            for i in range(val_arr.shape[0]):
                new_arr[i, :ind_arr[i]] = np.random.permutation(new_arr[i,:ind_arr[i]])
            return new_arr
        else:
            return val_arr


    def _permutation_augmentation(self, feed_dict):
        p = self.sess.run(self.augmentation_permutation)
        if p <= self.params["augmentation_min_permutation"]:
            return

        feed_dict[self.seq_word_left] = np.vstack([
            feed_dict[self.seq_word_left],
            self._permutation(feed_dict[self.seq_word_left], feed_dict[self.seq_len_word_left], p),
        ])
        feed_dict[self.seq_word_right] = np.vstack([
            feed_dict[self.seq_word_right],
            self._permutation(feed_dict[self.seq_word_right], feed_dict[self.seq_len_word_right], p),
        ])
        feed_dict[self.seq_char_left] = np.vstack([
            feed_dict[self.seq_char_left],
            self._permutation(feed_dict[self.seq_char_left], feed_dict[self.seq_len_char_left], p),
        ])
        feed_dict[self.seq_char_right] = np.vstack([
            feed_dict[self.seq_char_right],
            self._permutation(feed_dict[self.seq_char_right], feed_dict[self.seq_len_char_right], p),
        ])
        # double others
        feed_dict[self.seq_len_word_left] = np.tile(feed_dict[self.seq_len_word_left], 2)
        feed_dict[self.seq_len_word_right] = np.tile(feed_dict[self.seq_len_word_right], 2)
        feed_dict[self.seq_len_char_left] = np.tile(feed_dict[self.seq_len_char_left], 2)
        feed_dict[self.seq_len_char_right] = np.tile(feed_dict[self.seq_len_char_right], 2)
        feed_dict[self.labels] = np.tile(feed_dict[self.labels], 2)
        if self.params["use_features"]:
            feed_dict[self.features] = np.tile(feed_dict[self.features], [2, 1])


    def fit(self, X, Q, validation_data=None, shuffle=False, total_epoch=None):
        start_time = time.time()
        l = X["label"].shape[0]
        self.logger.info("fit on %d sample" % l)
        self.logger.info("max_batch: %d" % self.params["max_batch"])
        if validation_data is not None:
            self.logger.info("mean: %.5f"%np.mean(validation_data["label"]))
        train_idx_shuffle = np.arange(l)
        total_loss = 0.
        loss_decay = 0.9
        global_step = self.sess.run(self.global_step)
        if total_epoch is None:
            total_epoch = self.params["epoch"]
        for epoch in range(total_epoch):
            self.logger.info("epoch: %d" % (epoch + 1))
            np.random.seed(epoch)
            if shuffle:
                np.random.shuffle(train_idx_shuffle)
            dropout_p = self.sess.run(self.augmentation_dropout)
            batch_size = self.params["batch_size"]
            if dropout_p <= self.params["augmentation_min_dropout"]:
                batch_size *= 2
            batches = self._get_batch_index(train_idx_shuffle, batch_size)
            for i, idx in enumerate(batches):
                feed_dict = self._get_feed_dict(X, idx, Q, construct_neg=self.params["construct_neg"], training=True)
                # todo 这里为何要塞入这么多tf的节点？按理讲optimizer的输入应该是loss
                loss, lr, opt, summary, global_step = self.sess.run((self.loss, self.learning_rate, self.train_op, self.summary, self.global_step), feed_dict=feed_dict)
                self.train_writer.add_summary(summary, global_step)
                total_loss = loss_decay * total_loss + (1. - loss_decay) * loss
                if validation_data is not None and (self.params["eval_every_num_update"] > 0) and (global_step % self.params["eval_every_num_update"] == 0):
                    y_valid = validation_data["label"]
                    y_proba, y_proba_cal = self._predict_proba(validation_data, Q, fit_calibration=self.params["calibration"])
                    valid_loss = log_loss(y_valid, y_proba, eps=1e-15)
                    valid_loss_cal = log_loss(y_valid, y_proba_cal, eps=1e-15)
                    summary = tf.Summary()
                    summary.value.add(tag="logloss", simple_value=valid_loss)
                    self.test_writer.add_summary(summary, global_step)
                    self.logger.info(
                        "[epoch-%d, batch-%d] train-loss=%.5f, valid-loss=%.5f, valid-loss-cal=%.5f, valid-proba=%.5f, predict-proba=%.5f, predict-proba-cal=%.5f, lr=%.5f [%.1f s]" % (
                            epoch + 1, global_step, total_loss, valid_loss, valid_loss_cal,
                            np.mean(y_valid), np.mean(y_proba), np.mean(y_proba_cal), lr, time.time() - start_time))
                else:
                    self.logger.info("[epoch-%d, batch-%d] train-loss=%.5f, lr=%.5f [%.1f s]" % (
                        epoch + 1, global_step, total_loss,
                        lr, time.time() - start_time))
                if global_step >= self.params["max_batch"] and self.params["max_batch"] > 0:
                    break
            if global_step >= self.params["max_batch"] and self.params["max_batch"] > 0:
                break


    def _predict_node(self, X, Q, node):
        l = X["label"].shape[0]
        train_idx = np.arange(l)
        batches = self._get_batch_index(train_idx, self.params["batch_size"])
        y_pred = []
        y_pred_append = y_pred.append
        for idx in batches:
            feed_dict = self._get_feed_dict(X, idx, Q, training=False, symmetric=True)
            pred = self.sess.run(node, feed_dict=feed_dict)
            n = int(pred.shape[0]/2)
            pred = (pred[:n] + pred[n:])/2.
            y_pred_append(pred)
        y_pred = np.hstack(y_pred).reshape((-1, 1)).astype(np.float64)
        return y_pred


    def _predict_proba(self, X, Q, fit_calibration=False):
        y_logit = self._predict_node(X, Q, self.logits)
        y_proba = sigmoid(y_logit)
        y_proba_cal = y_proba
        if fit_calibration:
            y_valid = X["label"]
            self.calibration_model = LogisticRegression()
            self.calibration_model.fit(y_logit, y_valid)
        if self.calibration_model is not None:
            y_proba_cal = self.calibration_model.predict_proba(y_logit)[:,1]
        return y_proba, y_proba_cal


    def predict_proba(self, X, Q):
        _, y_proba_cal = self._predict_proba(X, Q, fit_calibration=False)
        return y_proba_cal


    def predict(self, X, Q):
        proba = self.predict_proba(X, Q)
        y = np.array(proba > self.threshold, dtype=int)
        return y
