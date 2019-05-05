"""
Minimal character-level Vanilla RNN model. Written by Andrej Karpathy (@karpathy)
BSD License
"""
import numpy as np

# data I/O
data = open('/home/blueberry/dl/e6/shakespeare_train.txt', 'r').read() # should be simple plain text file
chars = list(set(data)) #your code#  # 得到输入文件中所有字符种类 , 使用set去重
data_size, vocab_size =len(data),len(chars)  #your code##统计文件字符数和字符种类数
print ('data has %d characters, %d unique.' % (data_size, vocab_size))
char_to_ix = {}#your code# #构成从字母到数字的映射
ix_to_char = {}#your code# #构成数字到字母的映射
count = 0
for char in chars:
  char_to_ix[char] = count
  ix_to_char[count] = char
  count += 1



# hyperparameters
hidden_size = 100 # size of hidden layer of neurons
seq_length = 25 # number of steps to unroll the RNN for
learning_rate = 1e-1

# model parameters 初始化参数
Wxh = np.random.randn(hidden_size, vocab_size)*0.01 # input to hidden
Whh = np.random.randn(hidden_size, hidden_size)*0.01 # hidden to hidden
Why = np.random.randn(vocab_size, hidden_size)*0.01 # hidden to output
bh = np.zeros((hidden_size, 1)) # hidden bias
by = np.zeros((vocab_size, 1)) # output bias

def lossFun(inputs, targets, hprev):
  """
  inputs,targets are both list of integers.
  hprev is Hx1 array of initial hidden state
  returns the loss, gradients on model parameters, and last hidden state
  """
  xs, hs, ys, ps = {}, {}, {}, {}
  hs[-1] = np.copy(hprev)
  loss = 0
  # forward pass
  for t in range(len(inputs)):
    #encode inputs to 1-hot embedding,size(xs)=(len(input),vocab_size)
    xs[t] = np.zeros((vocab_size,1))#your code# # encode in 1-of-k representation 1-hot-encoding
    xs[t][inputs[t]] = 1 #your code# # encode in 1-of-k representation 1-hot-encoding
    #forward

    #hs[t] 是t时刻的hidden state， active function = np.tanh(z)，z = Wx*x_t+Wh*hs_(t-1) + bh,即本时刻输入层+一时刻个隐含层作为Z
    z = Wxh.dot(xs[t]) + Whh.dot(hs[t-1]) + bh
    hs[t] = np.tanh(z) #your code# # hidden state
    #ys[t] = w*hs[t]+by
    ys[t] = Why.dot(hs[t]) + by #your code# # unnormalized log probabilities for next chars
    #softmax(ys)
    ps[t] = np.exp(ys[t] - np.max(ys[t]))/np.sum(np.exp(ys[t] - np.max(ys[t]))) #your code# # probabilities for next chars
    #计算loss = cross_entropy（）
    loss += - np.log(ps[t][targets[t]]) #your code# # softmax (cross-entropy loss)
  # backward pass: compute gradients going backwards
  #初始化梯度
  dWxh, dWhh, dWhy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)
  dbh, dby = np.zeros_like(bh), np.zeros_like(by)
  dhnext = np.zeros_like(hs[0])
  for t in reversed(range(len(inputs))):
    #dy是softmax层求导，cross_entropy softmax 求导 aj-yi,yi为one-hot标签,aj为softmax之后第j个神经元输出，详情请见https://blog.csdn.net/u014313009/article/details/51045303
    dy =  ps[t]#your code#
    dy[targets[t]] -= 1 #your code# # backprop into y.
    #反向传播，求Why与by的导数
    dWhy += dy.dot(hs[t].T) #your code#
    dby += dy #your code#
    #反向传播到hidden state请参考https://blog.csdn.net/wjc1182511338/article/details/79191099完成，其中dh处反向传播的梯度外需加上dhnext
    dh = Why.T.dot(dy) + dhnext  #your code# # backprop into h

    dhraw =  dh * (1 - hs[t]**2) #your code# # backprop through tanh nonlinearity
    dbh += dhraw #your code#
    dWxh +=  dhraw .dot(xs[t].T)#your code#
    dWhh +=  dhraw. dot(hs[t-1].T) #your code#
    dhnext =  Whh.dot(dhraw) #your code#
  for dparam in [dWxh, dWhh, dWhy, dbh, dby]:
    np.clip(dparam, -5, 5, out=dparam) # clip to mitigate exploding gradients
  return loss, dWxh, dWhh, dWhy, dbh, dby, hs[len(inputs)-1]

def sample(h, seed_ix, n):
  """ 
  sample a sequence of integers from the model 
  h is memory state, seed_ix is seed letter for first time step
  """
  x = np.zeros((vocab_size, 1))
  x[seed_ix] = 1
  ixes = []
  for t in range(n):
    h = np.tanh(np.dot(Wxh, x) + np.dot(Whh, h) + bh)
    y = np.dot(Why, h) + by
    p = np.exp(y) / np.sum(np.exp(y))
    ix = np.random.choice(range(vocab_size), p=p.ravel())
    x = np.zeros((vocab_size, 1))
    x[ix] = 1
    ixes.append(ix)
  return ixes

n, p = 0, 0
mWxh, mWhh, mWhy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)
mbh, mby = np.zeros_like(bh), np.zeros_like(by) # memory variables for Adagrad
smooth_loss = -np.log(1.0/vocab_size)*seq_length # loss at iteration 0
while True:
  # prepare inputs (we're sweeping from left to right in steps seq_length long)
  if p+seq_length+1 >= len(data) or n == 0: 
    hprev = np.zeros((hidden_size,1)) # reset RNN memory
    p = 0 # go from start of data
  inputs = [char_to_ix[ch] for ch in data[p:p+seq_length]]
  targets = [char_to_ix[ch] for ch in data[p+1:p+seq_length+1]]

  # sample from the model now and then
  if n % 100 == 0:
    sample_ix = sample(hprev, inputs[0], 200)
    txt = ''.join(ix_to_char[ix] for ix in sample_ix)
    print ('----\n %s \n----' % (txt, ))

  # forward seq_length characters through the net and fetch gradient
  loss, dWxh, dWhh, dWhy, dbh, dby, hprev = lossFun(inputs, targets, hprev)
  smooth_loss = smooth_loss * 0.999 + loss * 0.001
  if n % 100 == 0: print ('iter %d, loss: %f' % (n, smooth_loss)) # print progress)
  
  # perform parameter update with Adagrad
  for param, dparam, mem in zip([Wxh, Whh, Why, bh, by], 
                                [dWxh, dWhh, dWhy, dbh, dby], 
                                [mWxh, mWhh, mWhy, mbh, mby]):
    mem += dparam * dparam
    param += -learning_rate * dparam / np.sqrt(mem + 1e-8) # adagrad update

  p += seq_length # move data pointer
  n += 1 # iteration counter 
