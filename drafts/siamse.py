import sys
import numpy as np
from sklearn.utils import shuffle
import tensorflow as tf
from tensorflow.python.framework import ops
sys.path.append("../spykesim")
import editsim
import seaborn as sns
sns.set_context('poster')
sns.set_style('white')
sns.set_color_codes()
plot_kwds = {'alpha' : 0.5, 's' : 80, 'linewidths':0}
ops.reset_default_graph()

class SNN(object):
    def __init__(self, n_in, n_hidden, n_embed, input_length):
        self.n_in = n_in
        self.n_hidden = n_hidden
        self.n_embed = n_embed
        self.input_length = input_length
        self._history = {
            'loss': []
        }
        
    def bi_rnn(self, x, forget_bias = 1.0, keep_prob = 0.8):
        cell_unit = tf.contrib.rnn.BasicLSTMCell
        # Forward direction cell
        lstm_forward_cell = cell_unit(self.n_hidden,
                                      forget_bias = forget_bias)
        lstm_forward_cell = tf.contrib.rnn.DropoutWrapper(lstm_forward_cell,
                                                          output_keep_prob = keep_prob)
        # Backward direction cell
        lstm_backward_cell = cell_unit(self.n_hidden,
                                       forget_bias = forget_bias)
        lstm_backward_cell = tf.contrib.rnn.DropoutWrapper(lstm_backward_cell,
                                                           output_keep_prob = keep_prob)
        # Split title into a character sequence
        #input_embed_split = tf.split(axis = 1,
        #                             num_or_size_splits = self.input_length, value = x)
        #input_embed_split = [tf.squeeze(input_, axis=[1]) for input_ in input_embed_split]
        #outputs, _, _ = tf.contrib.rnn.static_bidirectional_rnn(lstm_forward_cell,
        #                                                        lstm_backward_cell,
        #                                                        input_embed_split,
        #                                                        dtype=tf.float32)
        outputs, _, _ = tf.contrib.rnn.static_bidirectional_rnn(lstm_forward_cell,
                                                                lstm_backward_cell,
                                                                tf.unstack(x, axis = 1),
                                                                dtype=tf.float32)
        # Try to add another layer

        # Average The output over the sequence
        temporal_mean = tf.add_n(outputs) / self.input_length
        # Fully connected layer
        A = tf.get_variable(name="A", shape=[2 * self.n_hidden, self.n_embed],
                            dtype=tf.float32,
                            initializer=tf.random_normal_initializer(stddev=0.1))
        b = tf.get_variable(name="b", shape=[self.n_embed],
                            dtype=tf.float32,
                            initializer=tf.random_normal_initializer(stddev=0.1))        
        final_output = tf.matmul(temporal_mean, A) + b
        final_output = tf.nn.dropout(final_output, keep_prob)
        #temporal_mean = tf.reduce_mean(x, axis = 1)
        #A = tf.get_variable(name="A", shape=[20, self.n_embed],
        #                    dtype=tf.float32,
        #                    initializer=tf.random_normal_initializer(stddev=0.1))
        #b = tf.get_variable(name="b", shape=[self.n_embed],
        #                    dtype=tf.float32,
        #                    initializer=tf.random_normal_initializer(stddev=0.1))        
        #final_output = tf.matmul(temporal_mean, A) + b
        #final_output = tf.nn.dropout(final_output, keep_prob)
        return(final_output)

    def inference(self, x1, x2):
        output1 = self.bi_rnn(x1)
        # Declare that we will use the same variables on the second string
        with tf.variable_scope(tf.get_variable_scope(), reuse=True):
            output2 = self.bi_rnn(x2)
        self._bi_rnn = output1
        # Unit normalize the outputs
        output1 = tf.nn.l2_normalize(output1, 1)
        output2 = tf.nn.l2_normalize(output2, 1)
        # output1 = tf.sigmoid(output1)
        # output2 = tf.sigmoid(output2)
        # Return cosine distance
        #   in this case, the dot product of the norms is the same.
        dot_prod = tf.reduce_sum(tf.multiply(output1, output2), 1)
        return(dot_prod)

    def predict(self, X):
        return self._y.eval(session=self._sess,
                            feed_dict={self._x1: [x[0] for x in X], self._x2: [x[1] for x in X]})

    def embed(self, X):
        return self._bi_rnn.eval(session=self._sess,
                                 feed_dict={self._x1: [x for x in X]})

    #def get_predictions(scores):
    #    predictions = tf.sign(scores, name="predictions")
    def loss(self, scores, y_target, margin = 0.25):
        # Calculate the positive losses
        pos_loss_term = 0.25 * tf.square(tf.subtract(1., scores))
        
        # If y-target is -1 to 1, then do the following
        pos_mult = tf.add(tf.multiply(0.5, tf.cast(y_target, tf.float32)), 0.5)
        # Else if y-target is 0 to 1, then do the following
        pos_mult = tf.cast(y_target, tf.float32)
        
        # Make sure positive losses are on similar strings
        positive_loss = tf.multiply(pos_mult, pos_loss_term)
        
        # Calculate negative losses, then make sure on dissimilar strings
        
        # If y-target is -1 to 1, then do the following:
        neg_mult = tf.add(tf.multiply(-0.5, tf.cast(y_target, tf.float32)), 0.5)
        # Else if y-target is 0 to 1, then do the following
        neg_mult = tf.subtract(1., tf.cast(y_target, tf.float32))
        
        negative_loss = neg_mult*tf.square(scores)
        
        # Combine similar and dissimilar losses
        loss = tf.add(positive_loss, negative_loss)
        
        # Create the margin term.  This is when the targets are 0.,
        #  and the scores are less than m, return 0.
        
        # Check if target is zero (dissimilar strings)
        target_zero = tf.equal(tf.cast(y_target, tf.float32), 0.)
        # Check if cosine outputs is smaller than margin
        less_than_margin = tf.less(scores, margin)
        # Check if both are true
        both_logical = tf.logical_and(target_zero, less_than_margin)
        both_logical = tf.cast(both_logical, tf.float32)
        # If both are true, then multiply by (1-1)=0.
        multiplicative_factor = tf.cast(1. - both_logical, tf.float32)
        total_loss = tf.multiply(loss, multiplicative_factor)
        
        # Average loss over batch
        avg_loss = tf.reduce_mean(total_loss)
        return(avg_loss)

    # def loss(self, y, t):
    #     loss = tf.reduce_mean(tf.square(tf.subtract(t, y)))
    #     return loss

    def training(self, loss):
        optimizer = tf.train.AdamOptimizer(0.01)
        train_step = optimizer.minimize(loss)
        return train_step

    def fit(self, X_train, Y_train, X_test, Y_test,
            epochs=100, batch_size=100, n_batches = 100, p_keep=0.8, verbose=True, every = 10):
        nrow, ncol = X_train[0][0].shape
        x1 = tf.placeholder(tf.float32, [None, nrow, ncol], name="x1")
        x2 = tf.placeholder(tf.float32, [None, nrow, ncol], name="x2")
        t = tf.placeholder(tf.float32, [None], name="target")
        keep_prob = tf.placeholder(tf.float32, name="keep_prob")
        self._x1 = x1
        self._x2 = x2
        self._t = t
        self._keep_prob = keep_prob

        y = self.inference(x1, x2)
        loss = self.loss(y, t)
        train_step = self.training(loss)

        init = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init)

        self._y = y
        # For later evaluation
        self._sess = sess

        N_train = len(X_train)
        n_batches = N_train // batch_size

        for epoch in range(epochs):
            X_, Y_ = shuffle(X_train, Y_train)
            for i in range(n_batches):
                start = i * batch_size
                end = start + batch_size
                sess.run(train_step,
                         feed_dict={
                             x1: [x[0] for x in X_[start:end]],
                             x2: [x[1] for x in X_[start:end]],
                             t: Y_[start:end],
                             keep_prob: p_keep
                         })
            loss_ = loss.eval(session=sess,
                              feed_dict={
                                  x1: [x[0] for x in X_train],
                                  x2: [x[1] for x in X_train],
                                  t: Y_train,
                                  keep_prob: 1.0
                              })
            self._history["loss"].append(loss_)
            if verbose and epoch % every == 0:
                print(f"epoch: {epoch}, loss: {loss_}")
                loss_ = loss.eval(session=sess,
                                  feed_dict={
                                      x1: [x[0] for x in X_test],
                                      x2: [x[1] for x in X_test],
                                      t: Y_test,
                                      keep_prob: 1.0
                                  })
                print(f"test epoch: {epoch}, loss: {loss_}")

        return self._history

        

    def evaluate(self, X_test, Y_test):
        loss = self.loss(self._y, self._t)
        return loss.eval(session=self._sess,
                              feed_dict={
                                  self._x1: [x[0] for x in X_test],
                                  self._x2: [x[1] for x in X_test],
                                  self._t: Y_test,
                                  self._keep_prob: 1.0
                              })


#def gen_data(min_, max_, nrow, ncol):
#    return np.random.randint(
#        min_, max_, nrow * ncol).reshape(nrow, ncol)


# # X = np.array([
# #     [gen_data(min_, max_, nrow, ncol), gen_data(min_, max_, nrow, ncol)] for _ in range(n_train + n_test)
# # ])
# # Y = np.array([
# #     editsim.clocal_exp_editsim(xs[0].astype(np.double), xs[1].astype(np.double))[0] for xs in X
# # ])
# # X_mean = np.mean(X)
# # X_std = np.std(X)
# # X = (X - X_mean) / X_std
# # Y_mean = np.mean(Y)
# # Y_std = np.std(Y)
# # Y = (Y - Y_mean) / Y_std
# # from sklearn.model_selection import train_test_split
# # X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.33)
# 
# def sim1(x, y):
#     vec = np.zeros(x.shape[1])
#     for col in range(1, x.shape[1]):
#         vec[col] = np.dot(x[:, col] + x[:, col-1],  y[:, col] + y[:, col-1])
#     return vec.mean()
# X_train_ = np.array([
#     [gen_data(min_, max_, nrow, ncol), gen_data(min_, max_, nrow, ncol)] for _ in range(n_train)
# ])
# X_test_ = np.array([
#     [gen_data(min_, max_, nrow, ncol), gen_data(min_, max_, nrow, ncol)] for _ in range(n_test)
# ])
# #Y_train_ = np.array([
# #    editsim.clocal_exp_editsim(xs[0].astype(np.double), xs[1].astype(np.double))[0] for xs in X_train_
# #])
# #Y_test_ = np.array([
# #    editsim.clocal_exp_editsim(xs[0].astype(np.double), xs[1].astype(np.double))[0] for xs in X_test_
# #])
# Y_train_ = np.array([
#     sim1(xs[0].astype(np.double), xs[1].astype(np.double)) for xs in X_train_
# ])
# Y_test_ = np.array([
#     sim1(xs[0].astype(np.double), xs[1].astype(np.double)) for xs in X_test_
# ])
# xt_mean = np.mean(X_train_)
# xt_std = np.std(X_train_)
# yt_mean = np.mean(Y_train_)
# yt_std = np.std(Y_train_)
# 
# X_train = (X_train_ - xt_mean) / xt_std
# X_test = (X_test_ - xt_mean) / xt_std
# Y_train = (Y_train_ - yt_mean) / yt_std
# Y_test = (Y_test_ - yt_mean) / yt_std
# 
# #    def __init__(self, n_in, n_hidden, n_embed, input_length):
# snn = SNN(10, 100, 20, ncol)
# snn.fit(X_train, Y_train, X_test, Y_test, epochs = 100, batch_size = 300, p_keep = .8)
# print("Evaluate: ", snn.evaluate(X_test, Y_test))
# print("Evaluate: ", snn.evaluate(X_train, Y_train))
