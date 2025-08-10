from utilities import read_file_from
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from numpy.random import default_rng

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

class Neural_n_gram_model:

    def __init__(self, vocabulary_size: int):
        rng = np.random.default_rng(seed=42)  
        np_table = rng.normal(
            loc=0.0,          # mean
            scale=0.1,        # standard deviation
            size=(vocabulary_size, vocabulary_size)
        ).astype(np.float32)
        #np_table = 0,3 * rng.standard_normal((vocabulary_size, vocabulary_size)).astype(np.float32)
        #self.token_embedding_table = torch.from_numpy(np_table).requires_grad_()
        self.token_embedding_table = np_table


    #cross entropy is computed over unnormalized logits
    #only logit of the target class needs to be normalized
    #to compute cross entropy of the respective prediction
    def _get_cross_entropy (self, logits, target_index):
        softmax_denominator = np.sum(np.exp(logits))
        return -logits[target_index] + np.log(softmax_denominator)
    
    def _get_normalized_logits(self, logits):
        softmax_denominator = np.sum(np.exp(logits))
        return np.exp(logits) / softmax_denominator
    
    def forward(self, token_idx: int, target_index: int=None):
   
        logits = self.token_embedding_table[token_idx]
        #if the model is in the inference mode
        #just return the logits since
        #argumentum maximi of those is sufficient
        #to determine the next likeliest token 
        if target_index is None:  
            return logits
        
        #if the model is in training mode
        #just return the cross-entropy loss gradient
        #to accelerate gradient descent
        else:
            CEL_gradient = self._get_normalized_logits(logits)
            true_class_probability = CEL_gradient[target_index] 
            CEL_value = -np.log(true_class_probability)
            CEL_gradient[target_index] -= 1
            return CEL_value, CEL_gradient

   
    
    def generate_new_tokens(self, token_sequence: list, n_tokens: int, k_for_top_k:int = 4):
        for _ in range (n_tokens):
            last_token_index = token_sequence[-1]
          
            #next_token_index = np.argmax(self.token_embedding_table[last_token_index])
         
            #get the list of k likeliest next tokens
            topk = np.argpartition(self.token_embedding_table[last_token_index], -k_for_top_k)[-k_for_top_k:]

            #sample one token from k likeliest ones uniformly
            random_index = np.random.randint(0, k_for_top_k)  
            next_token_index = topk[random_index]

            token_sequence.append(next_token_index)
        return token_sequence






K = 2000
corpus_tokens = read_file_from("train", "tokenized_corpus", K)
vocabulary = read_file_from ("train", "learned_vocabularies", K)
vocabulary_size = len (vocabulary)

token_translator = Token_Translator(vocabulary)
encoded_tokens = token_translator.encode_list(corpus_tokens)
preceding_tokens = encoded_tokens[:-1]
subsequent_tokens = encoded_tokens[1:]


neural_n_gram_model = Neural_n_gram_model(vocabulary_size)




LEARNING_RATE = 0.4
LEARNING_RATE_MULTIPLIER_PER_EPOCH = 0.95
N_EPOCHS = 50




print (f"BPE k = {K}")
print (f"learning rate = {LEARNING_RATE}")
print (f"learning rate decay per epoch = {LEARNING_RATE_MULTIPLIER_PER_EPOCH}")
print (f"n_epochs = {N_EPOCHS}")
print ("_" * 50)


for epoch_index in range(1, N_EPOCHS + 1):

    print (f"current epoch number is {epoch_index}")
    print (f"current learning rate = {LEARNING_RATE}")
    total_loss = 0
    for x,y in zip(preceding_tokens, subsequent_tokens):
        #get the CEL gradient from the forward pass directly
        loss, loss_gradient = neural_n_gram_model.forward(x, y)
        #do gradient step immediately
        neural_n_gram_model.token_embedding_table[x] -= LEARNING_RATE*loss_gradient
        total_loss += loss
    
    LEARNING_RATE *= LEARNING_RATE_MULTIPLIER_PER_EPOCH

    total_loss /= len(preceding_tokens)
    print (f"average loss in epoch {epoch_index} is {total_loss}")

    input_sequence = [78]
    input_sequence = neural_n_gram_model.generate_new_tokens(input_sequence, 50)
    decoded_input_sequence = token_translator.decode_list(input_sequence)
    print ("".join(decoded_input_sequence).replace("</w>", " "))
    print ("_" * 50)
    
