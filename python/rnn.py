## Colin Vincent
## 12/7/17
## LSTM using tensorflow
## Adapted from stock price prediction example:
## https://lilianweng.github.io/lil-log/2017/07/08
##   /predict-stock-prices-using-RNN-part-1.html#overview-of-existing-tutorials
from RNNConfig import RNNConfig
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 

import tensorflow as tf 

## ------------ Variables ------------
input_size = 1
num_steps = 30
lstm_size=128
num_layers=1
keep_prob=0.8
batch_size=64
init_learning_rate = 0.001
learning_rate_decay = 0.99
init_epoch =5
max_epoch = 50

## ------------ Data ------------

## Read in the dataset
dataset = pd.read_csv('dataset/training.csv')

## Pull out the price data
price = dataset['price']

## Convert to numpy array
data = price.values
print "# of Price values: " + str(data.size)

## Plot price data vs time
plt.plot(data)
plt.show()

train_start = 0
train_end = data.size * .9
test_start = train_end
test_end = data.size

data_train = data[np.arange(train_start, train_end), :]
data_test = data[np.arange(test_start, test_end), :]

seq = [np.array(seq[i * input_size: (i + 1) * input_size]) 
       for i in range(len(seq) // input_size)]

# Split into groups of `num_steps`
X = np.array([seq[i: i + num_steps] for i in range(len(seq) - num_steps)])
y = np.array([seq[i + num_steps] for i in range(len(seq) - num_steps)])

## ------------ NN ------------

## Initialize graph
tf.reset_default_graph()
lstm_graph = tf.Graph()

with lstm_graph.as_default()

inputs = tf.placeholder(tf.float32, [None, num_steps, input_size])
targets = tf.placeholder(tf.float32, [None, input_size])
learning_rate = tf.placeholder(tf.float32, None)

cell = tf.contrib.rnn.MultiRNNCell(
        [_create_one_cell() for _ in range(config.num_layers)], 
        state_is_tuple=True
    ) if config.num_layers > 1 else _create_one_cell()

val, _ = tf.nn.dynamic_rnn(cell, inputs, dtype=tf.float32)

# Before transpose, val.get_shape() = (batch_size, num_steps, lstm_size)
# After transpose, val.get_shape() = (num_steps, batch_size, lstm_size)
val = tf.transpose(val, [1, 0, 2])
# last.get_shape() = (batch_size, lstm_size)
last = tf.gather(val, int(val.get_shape()[0]) - 1, name="last_lstm_output")

weight = tf.Variable(tf.truncated_normal([config.lstm_size, config.input_size]))
bias = tf.Variable(tf.constant(0.1, shape=[targets_width]))
prediction = tf.matmul(last, weight) + bias

loss = tf.reduce_mean(tf.square(prediction - targets))
optimizer = tf.train.RMSPropOptimizer(learning_rate)
minimize = optimizer.minimize(loss)

## ------------ Running Training session ------------
with tf.Session(graph=lstm_graph) as sess:

	## Initialize variables
	tf.global_variables_initializer().run()

	learning_rates = [
    	init_learning_rate * (config.learning_rate_decay ** max(float(i + 1 - config.init_epoch), 0.0)
    	) for i in range(config.max_epoch)]
        

        for epoch_step in range(max_epoch):

	        current_lr = learning_rates[epoch_step]
	        
	        # Check https://github.com/lilianweng/stock-rnn/blob/master/data_wrapper.py
	        # if you are curious to know what is StockDataSet and how generate_one_epoch() 
	        # is implemented.
	        for batch_X, batch_y in stock_dataset.generate_one_epoch(config.batch_size):
	            train_data_feed = {
	                inputs: batch_X, 
	                targets: batch_y, 
	                learning_rate: current_lr
	            }
	            train_loss, _ = sess.run([loss, minimize], train_data_feed)

saver = tf.train.Saver()
    saver.save(sess, "/trained_model", global_step=max_epoch_step)

## ------------ Helper Functions ------------

def generate_one_epoch(self, batch_size):
        num_batches = int(len(self.train_X)) // batch_size
        if batch_size * num_batches < len(self.train_X):
            num_batches += 1

        batch_indices = range(num_batches)
        random.shuffle(batch_indices)
        for j in batch_indices:
            batch_X = self.train_X[j * batch_size: (j + 1) * batch_size]
            batch_y = self.train_y[j * batch_size: (j + 1) * batch_size]
            assert set(map(len, batch_X)) == {self.num_steps}
yield batch_X, batch_y


## Returns an LSTM Cell depending on keep_prob value
def create_one_cell():

    return tf.contrib.rnn.LSTMCell(lstm_size, state_is_tuple=True)

    if keep_prob < 1.0:
        return tf.contrib.rnn.DropoutWrapper(lstm_cell, output_keep_prob=keep_prob)
