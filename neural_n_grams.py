import torch
import torch.nn.functional as F
import torch.optim as optim
from numpy.random import default_rng
from utilities import read_file_from

# -- 1) Load data and pre-encode once as Tensors
K = 2000
corpus_tokens  = read_file_from("train", "tokenized_corpus", K)
vocabulary     = read_file_from("train", "learned_vocabularies", K)
V              = len(vocabulary)
# map tokens to integer IDs
token_to_id    = {tok:i for i, tok in enumerate(vocabulary)}
preceding_ids  = [token_to_id[t] for t in corpus_tokens[:-1]]
subsequent_ids = [token_to_id[t] for t in corpus_tokens[1:]]

# convert lists → 1D LongTensors
ctx_ids = torch.tensor(preceding_ids,   dtype=torch.long)  # shape (N,)
tgt_ids = torch.tensor(subsequent_ids,  dtype=torch.long)  # shape (N,)
N       = ctx_ids.size(0)

# -- 2) Create a trainable bigram table (V×V) as before
table = torch.randn(V, V, dtype=torch.float32, requires_grad=True)

# -- 3) Set up optimizer
LEARNING_RATE = 0.005
optimizer = optim.Adam([table], lr=LEARNING_RATE)

# -- 4) Training hyperparams
EPOCHS     = 10
BATCH_SIZE = 128  # pick something that fits in memory

# -- 5) Batched training loop
for epoch in range(1, EPOCHS+1):
    # shuffle example order
    perm = torch.randperm(N)
    total_loss = 0.0

    # step through in batches
    for start in range(0, N, BATCH_SIZE):
        batch_idx = perm[start : start + BATCH_SIZE]  # shape (<=BATCH_SIZE,)

        # gather a batch of contexts and targets
        ctx_b = ctx_ids[batch_idx]   # shape (B,)
        tgt_b = tgt_ids[batch_idx]   # shape (B,)

        # forward: lookup logits for all B contexts → (B, V)
        logits_b = table[ctx_b]      # vectorized indexing

        # single call to cross_entropy on the batch
        loss = F.cross_entropy(logits_b, tgt_b)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * ctx_b.size(0)

    avg_loss = total_loss / N
    print(f"Epoch {epoch}/{EPOCHS}  avg loss = {avg_loss:.4f}")

# -- 6) (Optional) generation unchanged
def generate(table, start_id, length):
    seq = [start_id]
    for _ in range(length):
        nxt = table[seq[-1]].argmax().item()
        seq.append(nxt)
    return seq

def generate_sampling(table: torch.Tensor,
                      start_id: int,
                      length: int,
                      temperature: float = 1.0) -> list:
    """
    Generate a sequence of token-IDs by sampling from softmax(table[context]).
      table       – (V, V) Tensor where row i are logits for next-token|i
      start_id    – the initial context token‐ID (int)
      length      – how many new tokens to append
      temperature – >0, lower<1 makes distribution sharper, >1 makes it flatter
    Returns a Python list of length `1+length` of ints.
    """
    seq = [start_id]
    for _ in range(length):
        logits = table[seq[-1]]               # shape: (V,)
        scaled = logits / temperature         # apply temperature
        probs  = F.softmax(scaled, dim=0)     # (V,) sum to 1
        # sample one next_id from the categorical distribution
        next_id = torch.multinomial(probs, num_samples=1).item()
        seq.append(next_id)
    return seq


print (f"K = {K}")
print (f"batch size = {BATCH_SIZE}")
print (f"n_epochs = {EPOCHS}")
print(f"learning rate = {LEARNING_RATE}")
# Example generate from token 2

while True:
    try:
        integer = int(input("Enter an integer: "))
        out_ids    = generate(table, start_id=integer, length=50)
        out_tokens = [vocabulary[i] for i in out_ids]
        print("".join(out_tokens).replace("</w>", " "))
        out_ids    = generate_sampling(table, start_id=integer, length=50)
        out_tokens = [vocabulary[i] for i in out_ids]
        print("".join(out_tokens).replace("</w>", " "))
        
    except ValueError:
        print("Sorry, that wasn’t a valid integer, please try again.")

"""
import torch
import torch.nn.functional as F
import torch.optim as optim
from utilities import read_file_from

# 1) Load & encode once
K               = 400
raw_tokens      = read_file_from("train", "tokenized_corpus", K)
vocab           = read_file_from("train", "learned_vocabularies", K)
V               = len(vocab)

# build preceding / subsequent lists
preceding = raw_tokens[:-1]   # python list of length N-1
subsequent = raw_tokens[1:]

# 2) Pre-convert to 1D LongTensors
ctx_ids = torch.tensor([vocab.index(t) for t in preceding], dtype=torch.long)
tgt_ids = torch.tensor([vocab.index(t) for t in subsequent], dtype=torch.long)
N       = ctx_ids.size(0)

# 3) Create a trainable bigram‐table
table = torch.randn(V, V, dtype=torch.float32, requires_grad=True)

# 4) Optimizer
optimizer = optim.Adam([table], lr=1e-3)

# 5) Training loop (no list→Tensor in here)
EPOCHS = 5
for epoch in range(1, EPOCHS+1):
    total_loss = 0.0

    for i in range(N):
        # a) pick the i-th example
        ctx = ctx_ids[i]     # scalar LongTensor
        tgt = tgt_ids[i]     # scalar LongTensor

        # b) lookup logits for that single context
        #    yields a 1-D float Tensor of shape (V,)
        logits = table[ctx]          

        # c) turn into shape (1, V) and (1,) for cross_entropy
        logits_b = logits.unsqueeze(0)  # (1, V)
        tgt_b    = tgt.unsqueeze(0)     # (1,)

        # d) compute loss, backward, step
        loss = F.cross_entropy(logits_b, tgt_b)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg = total_loss / N
    print(f"Epoch {epoch:2d}/{EPOCHS}  avg loss = {avg:.4f}")

# 6) Generation (greedy)
def generate(table, start_id, length):
    seq = [start_id]
    for _ in range(length):
        nxt = table[seq[-1]].argmax().item()
        seq.append(nxt)
    return seq

# sample from token 2
out_ids = generate(table, start_id=2, length=50)
out_tokens = [vocab[i] for i in out_ids]
print("".join(out_tokens).replace("</w>", " "))
"""

"""
from utilities import read_file_from
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from numpy.random import default_rng

class Neural_n_gram_model:

    def __init__(self, vocabulary_size: int):
        rng = np.random.default_rng(seed=42)  
        np_table = rng.standard_normal((vocabulary_size, vocabulary_size)).astype(np.float32)
        self.token_embedding_table = torch.from_numpy(np_table).requires_grad_()
      

    
    def forward(self, token_idx: int, target: int=None):
        # 1) lookup, this yields a 1-D torch.Tensor of length C
        logits = self.token_embedding_table[token_idx]  # shape (C,)

        loss = None
        if target is not None:
            # 2) add batch-dim to logits → shape (1,C)
            logits = logits.unsqueeze(0)

            # 3) wrap the integer label into a LongTensor of shape (1,)
            #tgt = torch.tensor([target], dtype=torch.long)

            # 4) now input is (1,C) float and tgt is (1,) long → OK!
            loss = F.cross_entropy(logits, target)

        return logits, loss
    
    def generate_new_tokens(self, token_sequence: list, n_tokens: int):
        for _ in range (n_tokens):
            last_token_index = token_sequence[-1]
            next_token_index = torch.argmax(self.token_embedding_table[last_token_index])
            token_sequence.append(next_token_index)
        return token_sequence

class Token_Translator:
    def __init__(self, vocab: list):
        self.symbol_to_index = {symbol:index for index, symbol in enumerate(vocab)}
        self.index_to_symbol = {index:symbol for index, symbol in enumerate(vocab)}

    def encode_list(self, list_to_encode: list):
        return [self.symbol_to_index[i] for i in list_to_encode]
    
    def decode_list(self, list_to_decode):
        result = []
        for i in list_to_decode:
            # if i is a torch tensor, extract its value
            if isinstance(i, torch.Tensor):
                i = i.item()   # now a Python int
            result.append(self.index_to_symbol[i])
        return result




K = 400
corpus_tokens = read_file_from("train", "tokenized_corpus", K)
vocabulary = read_file_from ("train", "learned_vocabularies", K)
vocabulary_size = len (vocabulary)

token_translator = Token_Translator(vocabulary)
encoded_tokens = token_translator.encode_list(corpus_tokens)
preceding_tokens = encoded_tokens[:-1]
subsequent_tokens = encoded_tokens[1:]

context_indices = torch.tensor(preceding_tokens)
target_indices = torch.tensor(subsequent_tokens)

neural_n_gram_model = Neural_n_gram_model(vocabulary_size)




LEARNING_RATE = 0.001
N_EPOCHS = 15
optimizer= optim.Adam([neural_n_gram_model.token_embedding_table], lr=LEARNING_RATE)

for epoch_index in range(N_EPOCHS):
    #for x,y in zip(preceding_tokens, subsequent_tokens):
    for x,y in zip(context_indices, target_indices):
        predictions, loss = neural_n_gram_model.forward(x, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
  
    print (f"{epoch_index+1} epochs passed")
    input_sequence = [2]
    input_sequence = neural_n_gram_model.generate_new_tokens(input_sequence, 50)
    decoded_input_sequence = token_translator.decode_list(input_sequence)
    print ("".join(decoded_input_sequence).replace("</w>", " "))
    print ("__________________________________________________________")





print (f"k = {K}")
print (f"learning rate = {LEARNING_RATE}")
print (f"n_epochs = {N_EPOCHS}")

#print (f"n_iter = {N_ITER}")



print (type(neural_n_gram_model.token_embedding_table))


"""