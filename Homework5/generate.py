import numpy as np
import pickle

np.random.seed(11) ### 可复现性

a = np.load(open("char-rnn-snapshot.npz","rb"))
Wxh = a["Wxh"] 
Whh = a["Whh"]
Why = a["Why"]
bh = a["bh"]
by = a["by"]
mWxh, mWhh, mWhy = a["mWxh"], a["mWhh"], a["mWhy"]
mbh, mby = a["mbh"], a["mby"]
chars, data_size, vocab_size, char_to_ix, ix_to_char = a["chars"].tolist(), a["data_size"].tolist(), a["vocab_size"].tolist(), a["char_to_ix"].tolist(), a["ix_to_char"].tolist()
hidden_size = 250


def sample_softmax(ys):
    #softmax(ys)
    pt = np.exp(ys - np.max(ys))/np.sum(np.exp(ys - np.max(ys))) #your code# # probabilities for next chars
    ix = np.random.choice(range(vocab_size), p= pt.ravel())    
    return ix_to_char[ix]
    

def sample_gumble_softmax(ys,tempreature):
    noise = np.random.gumbel()
    ys = (noise + ys)/tempreature 
    pt = np.exp(ys - np.max(ys))/np.sum(np.exp(ys - np.max(ys))) #your code# # probabilities for next chars
    ix = np.random.choice(range(vocab_size), p= pt.ravel())
    return ix_to_char[ix]

def sample_single_word(last_word = "a",last_state = np.zeros((hidden_size,1))):
    xs = np.zeros((vocab_size,1))
    xs[char_to_ix[last_word]] = 1 

    #hs[t] 是t时刻的hidden state， active function = np.tanh(z)，z = Wx*x_t+Wh*hs_(t-1) + bh,即本时刻输入层+一时刻个隐含层作为Z
    z = Wxh.dot(xs) + Whh.dot(last_state) + bh
    hs = np.tanh(z) 
    ys = Why.dot(hs) + by 
    return ys,hs

def sample(mode = "softmax", tempreature = 1, last_hidden = np.zeros((hidden_size,1))):
    final_str = " "
    word = " "
    for i in range(400):
        if mode == "softmax":
            y,last_hidden = sample_single_word(word,last_hidden)
            word = sample_softmax(y)
        else:
            y,last_hidden = sample_single_word(word,last_hidden)            
            word = sample_gumble_softmax(y,tempreature)
        final_str += word
    print(final_str)

def get_last_hidden(file = "samples.txt"):
    content = open(file,"r+").read()
    inputs = [char_to_ix[c] for c in content if c in char_to_ix.keys()]
    h = np.zeros((hidden_size,1))
    for t in range(len(inputs)):
        x = np.zeros((vocab_size, 1))
        x[inputs[t]] = 1
        z = Wxh.dot(x) + Whh.dot(h) + bh
        h = np.tanh(z) 
    return h


h = get_last_hidden()
print("-"*30 + "softmax sample"+"-"*30)
sample(mode = "softmax" ,last_hidden = h)

print("-"*20 + "gumbel sample temp:1"+"-"*20)
sample(mode = "gumbel" ,tempreature = 1,last_hidden = h)


print("-"*20 + "gumbel sample temp:2"+"-"*20)
sample(mode = "gumbel" ,tempreature = 2,last_hidden = h)


print("-"*20 + "gumbel sample temp:5"+"-"*20)
sample(mode = "gumbel" ,tempreature = 5,last_hidden = h)

print("-"*20 + "gumbel sample temp:10"+"-"*20)
sample(mode = "gumbel" ,tempreature = 10,last_hidden = h)

print(char_to_ix[":"])