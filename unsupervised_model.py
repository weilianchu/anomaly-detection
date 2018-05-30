import sys
import json
import argparse
import tempfile
import pandas as pd
import operator
import subprocess
import os
import hashlib
from datetime import datetime
import numpy as np
import binascii
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

COLUMNS = []
LABEL_COLUMN = "label"
CATEGORICAL_COLUMNS = []
CONTINUOUS_COLUMNS = []

FLAGS = None


# Parameters
learning_rate = 0.001
training_epochs = 2000
batch_size = 200
display_step = 50

# Network Parameters
n_hidden_1 = 20 # 1st layer num features
n_hidden_2 = 10 # 2nd layer num features
n_input = None
data_dir = '.'


# Building the encoder
def encoder(x):
    # Encoder Hidden layer with sigmoid activation #1
    layer_1 = tf.nn.tanh(tf.add(tf.matmul(x, weights['encoder_h1']),
                                   biases['encoder_b1']))
    # Decoder Hidden layer with sigmoid activation #2
    layer_2 = tf.nn.tanh(tf.add(tf.matmul(layer_1, weights['encoder_h2']),
                                   biases['encoder_b2']))
    return layer_2


# Building the decoder
def decoder(x):
    # Encoder Hidden layer with sigmoid activation #1
    layer_1 = tf.nn.tanh(tf.add(tf.matmul(x, weights['decoder_h1']),
                                   biases['decoder_b1']))
    # Decoder Hidden layer with sigmoid activation #2
    layer_2 = tf.nn.tanh(tf.add(tf.matmul(layer_1, weights['decoder_h2']),
                                   biases['decoder_b2']))
    return layer_2


def df_to_pcap(j, df_predict, file):
    linux_cooked_header = df_predict.at[j, 'linux_cooked_header'];
    frame_raw = df_predict.at[j, 'frame_raw']
    # for Linux cooked header replace dest MAC and remove two bytes to reconstruct normal frame using text2pcap
    if (linux_cooked_header):
        frame_raw = "000000000000" + frame_raw[6*2:] # replce dest MAC
        frame_raw = frame_raw[:12*2] + "" + frame_raw[14*2:] # remove two bytes before Protocol
    hex_to_txt(frame_raw, file)

    
def to_pcap_file(filename, output_pcap_file):
    FNULL = open(os.devnull, 'w')
    subprocess.call(["text2pcap", filename, output_pcap_file], stdout=FNULL, stderr=subprocess.STDOUT)

    
def hex_to_txt(hexstring, output_file):
    h = hexstring.lower()
    
    file = open(output_file, 'a')
    
    for i in range(0, len(h), 2):
        if(i%32 == 0):
            file.write(format(i/2, '06x') + ' ')
        
        file.write(h[i:i+2] + ' ')
        
        if(i%32 == 30):
            file.write('\n')

    file.write('\n')
    file.close()

    
def json_collector(dict, name):
    r = []
    if hasattr(dict, 'items'):
        for k, v in dict.items():
            if (k in name):
                r.append(v)
            else:
                val = json_collector(v, name)
                if (len(val) > 0):
                    r = r + val

    return r
   
    
def readJsonEKLine(df, line, label):
    # trim end of lines
    line = line.rstrip('\n')
    # skip empty lines
    if (line.rstrip() == ""):
        return

    try:
        j = json.loads(line)
    except:
        print "this line ended early"
        return
                
    # frames
    if ('layers' in j):
        layers = j['layers']
        
        linux_cooked_header = False
        if ('sll_raw' in layers):
            linux_cooked_header = True
        if ('frame_raw' in layers):
            
            i = len(df)
            
            df.loc[i, 'frame_raw'] = layers['frame_raw']
            df.loc[i, 'linux_cooked_header'] = linux_cooked_header
            
            for c in COLUMNS:
                v = json_collector(j, [c])
                if (len(v) > 0):
                    v = v[0]
                else:
                    v = ''
                df.loc[i, c] = v
                
            df.loc[i, 'label'] = label
            

def readJsonEK(df, filename, label, limit = 0):
    i = 0
    while i <= limit:
        with open(filename) as f:
            for line in f:
                if (limit != 0 and i > limit):
                    return i
                readJsonEKLine(df, line, label)
                i = i + 1
    return i

global COLUMNS
global CATEGORICAL_COLUMNS

# only raw json fielads are accepted
COLUMNS = ["ip_raw", "tcp_tcp_srcport_raw", "tcp_tcp_dstport_raw"]#["ip_ip_src_raw", "ip_ip_dst_raw"]
# COLUMNS =  ["ip_ip_src_raw",
#             "ip_ip_dst_raw",
#             "tcp_tcp_srcport_raw",
#             "tcp_tcp_dstport_raw",
#             "eth_dst_eth_dst_resolved_raw",
#             "eth_src_eth_addr_resolved_raw"]

CATEGORICAL_COLUMNS = COLUMNS

print COLUMNS
print CATEGORICAL_COLUMNS
print CONTINUOUS_COLUMNS

df = pd.DataFrame()

ln = readJsonEK(df, "MIT/test.json", 0)
#readJsonEK(df, FLAGS.anomaly_tshark_ek_x_json, 1, ln)

df = df.sample(frac=1).reset_index(drop=True)

print(df)

#####################################
# train neural network and evaluate #
#####################################
model_dir = tempfile.mkdtemp()
print("model directory = %s" % model_dir)

print df.columns

v_len = 0
col_max_len = []
for i in range(len(df.columns)):
    if (df.columns[i] in COLUMNS):
        m = df[df.columns[i]].map(len).max() / 2
        col_max_len.append(m)
        if (df.columns[i] != 'timestamp'):
            v_len = v_len + m
        else:
            min_t = int(df[df.columns[i]].min())
            max_t = int(df[df.columns[i]].max())
            print min_t
            print max_t
            v_len = v_len + 1
        
    else:
        col_max_len.append(None)

print col_max_len
print v_len

train_x = np.array([0]*(v_len), np.float)
#print train_x

for index, row in df.iterrows():
    #train_x = np.append(train_x, [1])
    
    f_array = []
    #print row
    c = 0
    for column in df.columns:
        if column in COLUMNS:
            #print i
            #print s
            #print s
            #print column
            v = row.loc[column]
            r = 0
            #print v
            if (column == 'timestamp'):
                v = int(v)
                #print v
                r = float(v - min_t)/float(max_t - min_t)
                #print r
                f_array.append(r)
            else:
                s = binascii.unhexlify(v)
                #print s
                for i in range(col_max_len[c]):
                    r = 0.0
                    if (i < len(s)):
                        #print ord(s[i])
                        r = float(ord(s[i]))/255.0
                    f_array.append(r)

        c = c + 1
    
    #print train_x
    #print f_array
    train_x = np.vstack([train_x, f_array])

# delete the first dummy row
train_x = np.delete(train_x, 0, 0)

print train_x
#print train_x.shape

mydf = pd.DataFrame(train_x)
print mydf.shape

# remove static columns
mydf = mydf.loc[:, (mydf != mydf.iloc[0]).any()]

#print mydf
print mydf.shape

plt.figure(figsize=(12,5*4))
gs = gridspec.GridSpec(len(mydf.columns), 1)
for i, cn in enumerate(mydf.columns):
    ax = plt.subplot(gs[i])
    sns.distplot(mydf[cn], bins=50)
    #sns.distplot(mydf[cn][mydf.Class == 0], bins=50)
    ax.set_xlabel('')
    ax.set_title('histogram of feature: ' + str(cn))
plt.show()

n_input = train_x.shape[1]
print "n_input = " + str(n_input)
print "n_input_1 = " + str(n_hidden_1)

X = tf.placeholder("float", [None, n_input])

weights = {
    'encoder_h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'encoder_h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    #'decoder_h1': tf.Variable(tf.random_normal([n_hidden_1, n_input])),
    'decoder_h1': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_1])),
    'decoder_h2': tf.Variable(tf.random_normal([n_hidden_1, n_input])),
}
biases = {
    'encoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'encoder_b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'decoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'decoder_b2': tf.Variable(tf.random_normal([n_input])),
}

encoder_op = encoder(X)
decoder_op = decoder(encoder_op)

# Prediction
y_pred = decoder_op
# Targets (Labels) are the input data.
y_true = X

# Define batch mse
batch_mse = tf.reduce_mean(tf.pow(y_true - y_pred, 2), 1)

# Define loss and optimizer, minimize the squared error
cost = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(cost)

save_model = os.path.join(data_dir, 'model.ckpt')
saver = tf.train.Saver()

# Initializing the variables
init = tf.global_variables_initializer()

c_hist = []

with tf.Session() as sess:
    now = datetime.now()
    sess.run(init)
    total_batch = int(train_x.shape[0]/batch_size)
    print "Total batches = " + str(total_batch)
    
    # Training cycle
    for epoch in range(training_epochs):
        # Loop over all batches
        for i in range(total_batch):
            batch_idx = np.random.choice(train_x.shape[0], batch_size)
            batch_xs = train_x[batch_idx]
            #print batch_xs.shape
            #print train_x.shape
            
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([optimizer, cost], feed_dict={X: batch_xs})

        # Display logs per epoch step
        if epoch % display_step == 0:
            train_batch_mse = sess.run(batch_mse, feed_dict={X: train_x})
            print("Epoch:", '%04d' % (epoch+1),
                  "cost=", "{:.9f}".format(c))
        c_hist.append(c)

    print("Optimization Finished!")
    
    #print c_hist
    plt.plot(c_hist)
    plt.title('model cost')
    plt.ylabel('cost')
    plt.xlabel('epoch')
    plt.legend(['train_x'], loc='upper right');
    plt.show()
    

    save_path = saver.save(sess, save_model)
    print("Model saved in file: %s" % save_path)
    
    #print train_batch_mse

save_model = os.path.join(data_dir, 'model.ckpt')
saver = tf.train.Saver()

# Initializing the variables
init = tf.global_variables_initializer()

with tf.Session() as sess:
    now = datetime.now()
    
    saver.restore(sess, save_model)
    
    test_batch_mse, reconstructed_x = sess.run([batch_mse, decoder_op], feed_dict={X: train_x})
    
    print("Epoch:", '%04d' % (epoch+1),
                  "cost=", "{:.9f}".format(c))
    
    print reconstructed_x
    print test_batch_mse

plt.hist(test_batch_mse, bins = 100)
plt.title("test_batch_mse mse distribution")
plt.xlabel("mse score")
plt.show()

mydf = pd.DataFrame(reconstructed_x)
mydf_orig = pd.DataFrame(train_x)
print mydf.shape
print mydf_orig.shape

# remove static columns
mydf = mydf.loc[:, (mydf != mydf.iloc[0]).any()]
mydf_orig = mydf_orig.loc[:, (mydf_orig != mydf_orig.iloc[0]).any()]

print mydf.shape
print mydf_orig.shape

plt.figure(figsize=(12,5*4))
gs = gridspec.GridSpec(len(mydf.columns), 1)
for i, cn in enumerate(mydf_orig.columns):
    ax = plt.subplot(gs[i])
    sns.distplot(mydf_orig[cn], bins=50)
    sns.distplot(mydf[cn], bins=50)
    #sns.distplot(mydf[cn][mydf.Class == 0], bins=50)
    ax.set_xlabel('')
    ax.set_title('histogram of feature: ' + str(cn))
plt.show()

# Generate pcap
# open TMP file used by text2pcap
infile = 'ad_test'
file = infile + '.tmp'
f = open(file, 'w')


print type(test_batch_mse)

# get top N of mse
N = 30
top_N = test_batch_mse.argsort()[-N:][::-1]
print top_N

for j in top_N:
    #print mse
    print("MSE = " + str(test_batch_mse[j]))
    print(str(df.iloc[[j]]))
    # pcap
    df_to_pcap(j, df, file)

    
# pcap
f.close()
to_pcap_file(infile + '.tmp', infile + '.pcap')
os.remove(infile + '.tmp')
print("Generated " + infile + ".pcap")

print("done")