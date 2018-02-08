import tensorflow as tf
import numpy as np
import utils
import data
import random
import argparse
import sys

BATCH_SIZE = 256

class LSTM_rnn():

    def __init__(self, state_size, num_classes,
            ckpt_path='ckpt/lstm/',
            model_name='lstm'):

	self.ckpt_path = ckpt_path
        self.model_name = model_name
        self.state_size = state_size
        self.num_classes = num_classes
        
        # build graph ops
        def __graph__():
            tf.reset_default_graph()
            # inputs fed to graph
            xs_ = tf.placeholder(shape=[None, None], dtype=tf.int32)
            ys_ = tf.placeholder(shape=[None], dtype=tf.int32)
            #
            # inputs are transformed into embeddings
            embs = tf.get_variable('emb', [num_classes, state_size])	
	    #embedding_lookup is a wrapper that basically selects a row of embs for each index in xs_, which is an array of indices.
            rnn_inputs = tf.nn.embedding_lookup(embs, xs_)   
            #
            # initial hidden state
            init_state = tf.placeholder(shape=[2, None, state_size], dtype=tf.float32, name='initial_state')
            # Returns an initializer performing "Xavier" initialization for weights.
            xav_init = tf.contrib.layers.xavier_initializer
            # params
            W = tf.get_variable('W', shape=[4, self.state_size, self.state_size], initializer=xav_init())
            U = tf.get_variable('U', shape=[4, self.state_size, self.state_size], initializer=xav_init())
            #b = tf.get_variable('b', shape=[self.state_size], initializer=tf.constant_initializer(0.))
            ####
            # step - LSTM
            def step(prev, x):
                # stacks previous internal state and output state
                st_1, ct_1 = tf.unstack(prev)
                # GATES
                #  input gate
                i = tf.sigmoid(tf.matmul(x,U[0]) + tf.matmul(st_1,W[0]))
                #  forget gate
                f = tf.sigmoid(tf.matmul(x,U[1]) + tf.matmul(st_1,W[1]))
                #  output gate
                o = tf.sigmoid(tf.matmul(x,U[2]) + tf.matmul(st_1,W[2]))
                #  gate weights
                g = tf.tanh(tf.matmul(x,U[3]) + tf.matmul(st_1,W[3]))
                # new internal cell state
                ct = ct_1*f + g*i
                # output state
                st = tf.tanh(ct)*o
                return tf.stack([st, ct])
            ###
            #calls the function fn for every entry in elem. fn is called for len(elem) times
            #   tf.scan(fn, elems, initializer)
            print("rnn_inputs",rnn_inputs)
            states = tf.scan(step, 
                    tf.transpose(rnn_inputs, [1,0,2]),
                    initializer=init_state)
            #The tensor of form [batch_size, seqlen, state_size], is transposed to [seqlen, batch_size, state_size].
            # predictions
            V = tf.get_variable('V', shape=[state_size, num_classes], 
                                initializer=xav_init())
            bo = tf.get_variable('bo', shape=[num_classes], 
                                 initializer=tf.constant_initializer(0.))

            ####
            # get last state before reshape/transpose
            last_state = states[-1]

            ####
            # transpose
            states = tf.transpose(states, [1,2,0,3])[0]
            #st_shp = tf.shape(states)
            # flatten states to 2d matrix for matmult with V
            #states_reshaped = tf.reshape(states, [st_shp[0] * st_shp[1], st_shp[2]])
            states_reshaped = tf.reshape(states, [-1, state_size])
            logits = tf.matmul(states_reshaped, V) + bo
            # predictions
            predictions = tf.nn.softmax(logits) 
            #
            # optimization
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits = logits, labels = ys_)
            loss = tf.reduce_mean(losses)
	    #does forward and backward propogation iteratively untill the loss is minimized
            train_op = tf.train.AdagradOptimizer(learning_rate=0.1).minimize(loss)
            #
            # expose symbols
            self.xs_ = xs_
            self.ys_ = ys_
            self.loss = loss
            self.train_op = train_op
            self.predictions = predictions
            self.last_state = last_state
            self.init_state = init_state
        ##### 
        # build graph
        sys.stdout.write('\n<log> Building Graph...')
        __graph__()
        sys.stdout.write('</log>\n')

    ####
    # training
    def train(self, train_set, epochs=100):
        # training session
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            train_loss = 0
            try:
                for i in range(epochs):
                    for j in range(100):
                        xs, ys = train_set.next()
			print("xs",xs)
			print("ys",ys)
                        batch_size = xs.shape[0]
                        _, train_loss_ = sess.run([self.train_op, self.loss], feed_dict = {
                                self.xs_ : xs,
                                self.ys_ : ys.flatten(),
                                self.init_state : np.zeros([2, batch_size, self.state_size])
                            })
                        train_loss += train_loss_
                    print('[{}] loss : {}'.format(i,train_loss/100))
                    train_loss = 0
            except KeyboardInterrupt:
                print('interrupted by user at ' + str(i))
            #
            # training ends here; 
            #  save checkpoint
            saver = tf.train.Saver()
            saver.save(sess, self.ckpt_path + self.model_name, global_step=i)
    ####
    # generate characters
    def generate(self, idx2w, w2idx, num_words=100, separator=' '):
        #
        # generate text
        random_init_word = random.choice(idx2w)
        current_word = w2idx[random_init_word]
        #
        # start session
        with tf.Session() as sess:
            # init session
            sess.run(tf.global_variables_initializer())
            #
            # restore session
            ckpt = tf.train.get_checkpoint_state(self.ckpt_path)
            saver = tf.train.Saver()
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
            # generate operation
            words = [current_word]
            state = None
            # enter the loop
            for i in range(num_words):
                if state:
                    feed_dict = {self.xs_ : np.array([current_word]).reshape([1,1]),
                            self.init_state : state_}
                else:
                    feed_dict = {self.xs_ : np.array([current_word]).reshape([1,1]),
                            self.init_state : np.zeros([2, 1, self.state_size])}
                #
                # forward propagation
                preds, state_ = sess.run([self.predictions, self.last_state], feed_dict=feed_dict)
                # 
                # set flag to true
                state = True
                # 
                # set new word
                current_word = np.random.choice(preds.shape[-1], 1, p=np.squeeze(preds))[0]
                # add to list of words
                words.append(current_word)
        ########
        # return the list of words as string
        return separator.join([idx2w[w] for w in words])

### 
# parse arguments
def parse_args():
    parser = argparse.ArgumentParser(
        description='Long Short Term Memory RNN for Text Hallucination, built with tf.scan')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-g', '--generate', action='store_true',
                        help='generate text')
    group.add_argument('-t', '--train', action='store_true',
                        help='train model')
    parser.add_argument('-n', '--num_words', required=False, type=int,
                        help='number of words to generate')
    args = vars(parser.parse_args())
    return args


###
# main function
if __name__ == '__main__':
    # parse arguments
    args = parse_args()
    #
    # fetch data
    X, Y, idx2w, w2idx = data.load_data('data/paulg/')  ##X (input) Y (output) array of indices of symbols/alphabets
    #idx2w : unique symbols list     w2idx: random indices associated to each symbols {' ': 0, '$': 1,}
    seqlen = X.shape[0]
    #
    # create the model
    model = LSTM_rnn(state_size = 512, num_classes=len(idx2w))   #num of classes are the total number of alphabets and symbols which are predicted
    # to train or to generate?
    if args['train']:
        # get train set
        train_set = utils.rand_batch_gen(X, Y ,batch_size=BATCH_SIZE) ##returns random sets of batchsize (256 sets) of X[sampleid] and y[sampleid]
        #
        # start training
        model.train(train_set)
    elif args['generate']:
        # call generate method
        text = model.generate(idx2w, w2idx, 
                num_words=args['num_words'] if args['num_words'] else 100,
                separator='')
        #########
        # text generation complete
        #
        print('______Generated Text_______')
        print(text)
        print('___________________________')
