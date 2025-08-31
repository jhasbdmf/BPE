from utilities import read_file_from, log_message
import numpy as np
import random
import copy
import matplotlib.pyplot as plt

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
            #if isinstance(i, torch.Tensor):
            #    i = i.item()   # now a Python int
            result.append(self.index_to_symbol[i])
        return result

class Neural_n_gram_model:

    def __init__(self, vocabulary_size: int):
        rng = np.random.default_rng(seed=42)  
        np_table = rng.normal(
            loc=0.0,          # mean
            scale=0.2,        # standard deviation
            size=(vocabulary_size, vocabulary_size)
        ).astype(np.float32)
        #np_table = 0,3 * rng.standard_normal((vocabulary_size, vocabulary_size)).astype(np.float32)
        #self.token_embedding_table = torch.from_numpy(np_table).requires_grad_()
        self.token_embedding_table = np_table


    #cross entropy is computed over unnormalized logits
    #only logit of the target class needs to be normalized
    #to compute cross entropy of the respective prediction
    def _get_cross_entropy (self, logits, target_index):
        softmax_denominator = np.sum(np.exp(logits)) + 1e-8
        return -logits[target_index] + np.log(softmax_denominator)
    
    def _get_normalized_logits(self, logits):
        softmax_denominator = np.sum(np.exp(logits))
        return np.exp(logits) / softmax_denominator
    
    def _get_normalized_logits_with_softmax_denom(self, logits):
        softmax_denominator = np.sum(np.exp(logits))
        return np.exp(logits) / softmax_denominator, softmax_denominator
    
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
            #CEL_gradient = self._get_normalized_logits(logits)
            normalized_logits, softmax_denom = self._get_normalized_logits_with_softmax_denom(logits)

            CEL_value = -logits[target_index] + np.log(softmax_denom)

            normalized_logits[target_index] -= 1
            CEL_gradient = normalized_logits

            #predicted_true_class_probability = normalized_logits[target_index] 
            #CEL_value = -np.log(predicted_true_class_probability)
            
            return CEL_value, CEL_gradient

   
    
    def generate_new_tokens(self, token_sequence: list, n_tokens: int, k_for_top_k:int = 3):
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
    
def evaluate_model_on(model, dataset):
    total_loss = 0
    for x, y in dataset:
        total_loss += model.forward(x, y)[0]
    return total_loss/len(dataset)


def train_model_with (model: Neural_n_gram_model, 
                         optimizer: str, 
                         lr: float, 
                         n_epochs: int, 
                         train_set,
                         val_set,
                         rho: float = 0.9,
                         sgd_lr_multiplier: float = 0.95
                        ):
    #token_pairs = list(zip(preceding_tokens, subsequent_tokens))
    train_loss_history = []
    val_loss_history = []

    best_avg_epoch_loss = 1000

    if optimizer.lower() == "rmsprop":
        RMSprop_running_square_gradient_avg = np.zeros_like(neural_n_gram_model_RMSprop.token_embedding_table)

    for epoch_index in range(1, n_epochs + 1):

        print (f"Epoch {epoch_index}/{n_epochs}")

        
        if optimizer.lower() == "rmsprop": 
            total_RMS_prop_lr = 0.0
        else:
            print (f"current SGD learning rate = {lr}")

        
        total_loss = 0
       
        
        #random.shuffle(token_pairs)
        random.shuffle(train_set) 
      
        #for x,y in token_pairs:
        for x,y in train_set:

            #get the CEL gradient from the forward pass directly
            loss, loss_gradient = model.forward(x, y)
            


            #compute running gradient avg for RMSprop
            #and do gradient descent step 
            if optimizer.lower() == "rmsprop":  
                RMSprop_running_square_gradient_avg[x] *=  rho
                RMSprop_running_square_gradient_avg[x] += (loss_gradient ** 2) * (1-rho)
                RMSprop_denominator = np.sqrt(RMSprop_running_square_gradient_avg[x] + 0.1e-6)
                current_lr = lr / RMSprop_denominator
                total_RMS_prop_lr += current_lr.mean()
                model.token_embedding_table[x] -=  current_lr * loss_gradient
            else:

            #just do gradient descent step for SGD
                model.token_embedding_table[x] -= lr*loss_gradient

        

            total_loss += loss
          
        if optimizer.lower() == "rmsprop":
            total_RMS_prop_lr /= len(preceding_tokens)
            print (f"current avg RMSprop learning rate = {total_RMS_prop_lr}")
        else:
            lr *= sgd_lr_multiplier
        
        

        #total_loss /= len(preceding_tokens)
        total_loss /= len(train_set)
        print (f"average train loss is {total_loss}")
        train_loss_history.append(total_loss)

        avg_val_loss = evaluate_model_on (model, val_set)
        print (f"average val loss = {avg_val_loss:.5f}")
        #log_message (f"average val loss = {avg_val_loss:.5f}")
        val_loss_history.append(avg_val_loss)

        if avg_val_loss < best_avg_epoch_loss:
            best_avg_epoch_loss = avg_val_loss
            best_model = copy.deepcopy(model)

        print ("_" * 50)

    if best_model is not None:
        return best_model, train_loss_history, val_loss_history
    else:
        return model, train_loss_history, val_loss_history
    #return model, train_loss_history, val_loss_history


def plot_perplexity (train_perplexity_scores1: list, 
                     val_perplexity_scores1: list, 
                     train_perplexity_scores2: list, 
                     val_perplexity_scores2: list, 
                     ):
    # Indices
    #indices1 = range(len(perplexity_scores_K_2000))  
    indices1 = np.arange(len(train_perplexity_scores1)) + 1
    indices2 = np.arange(len(val_perplexity_scores1)) + 1
    indices3 = np.arange(len(train_perplexity_scores2)) + 1
    indices4 = np.arange(len(val_perplexity_scores2)) + 1


    #indices2 = range(len(val_loss_history)) 

    # Plot both
    plt.plot(indices1, train_perplexity_scores1, marker='o', linestyle='-', label=f'Train perplexity RMSprop')
    plt.plot(indices2, val_perplexity_scores1, marker='s', linestyle='--', label=f'Val perplexity RMSprop')
    plt.plot(indices3, train_perplexity_scores2, marker='^', linestyle='-', label=f'Train perplexity SGD')
    plt.plot(indices4, val_perplexity_scores2, marker='p', linestyle='--', label=f'Val perplexity SGD')
  
    plt.xlabel('Epochs')
    plt.ylabel('Average perplexity')
    plt.title(f'Perplexity scores of neural bigrams')
    plt.legend()
    plt.grid(True)

    plt.savefig(f"Perplexity_scores_of_neural_bigrams.png", dpi=300, bbox_inches='tight')

    plt.show()


K = 400


corpus_tokens = read_file_from("train", "tokenized_corpus", K)
vocabulary = read_file_from ("train", "learned_vocabularies", K)
vocabulary_size = len (vocabulary)

token_translator = Token_Translator(vocabulary)
encoded_tokens = token_translator.encode_list(corpus_tokens)
preceding_tokens = encoded_tokens[:-1]
subsequent_tokens = encoded_tokens[1:]



train_set_400 = list(zip(preceding_tokens, subsequent_tokens))

val_tokens = read_file_from("valid", "tokenized_corpus", K)
encoded_tokens = token_translator.encode_list(val_tokens)
preceding_tokens = encoded_tokens[:-1]
subsequent_tokens = encoded_tokens[1:]

val_set_400 = list(zip(preceding_tokens, subsequent_tokens))

test_tokens = read_file_from("test", "tokenized_corpus", K)
encoded_tokens = token_translator.encode_list(test_tokens)
preceding_tokens = encoded_tokens[:-1]
subsequent_tokens = encoded_tokens[1:]

test_set_400 = list(zip(preceding_tokens, subsequent_tokens))





neural_n_gram_model_SGD = Neural_n_gram_model(vocabulary_size)
neural_n_gram_model_RMSprop = Neural_n_gram_model(vocabulary_size)



SGD_LEARNING_RATE = 0.5
RMS_PROP_INITIAL_LR = 0.3
LEARNING_RATE_MULTIPLIER_PER_EPOCH = 0.95
N_EPOCHS = 15

RMS_PROP_RHO = 0.9
#RMS_PROP_EPSILON = 0.1e-7



print (f"BPE k = {K}")
print (f"learning rate = {SGD_LEARNING_RATE}")
print (f"learning rate decay per epoch = {LEARNING_RATE_MULTIPLIER_PER_EPOCH}")
print (f"n_epochs = {N_EPOCHS}")
print ("_" * 50)




neural_n_gram_model_RMSprop, train_loss_history_RMSprop, val_loss_history_RMSprop = train_model_with (neural_n_gram_model_RMSprop,
                                                                      "RMSprop",
                                                                      RMS_PROP_INITIAL_LR,
                                                                      N_EPOCHS,
                                                                      train_set_400,
                                                                      val_set_400,
                                                                      RMS_PROP_RHO
                                                                      )


#print ("ASKJDHA", evaluate_model_on(neural_n_gram_model_RMSprop, val_set_400))

input_sequence_RMSprop = [78]
input_sequence_RMSprop = neural_n_gram_model_RMSprop.generate_new_tokens(input_sequence_RMSprop, 200)
decoded_input_sequence_RMSprop = token_translator.decode_list(input_sequence_RMSprop)
print ("RMSprop text")
RMS_text = "".join(decoded_input_sequence_RMSprop).replace("</w>", " ")
print (RMS_text)
print ("CEL", train_loss_history_RMSprop)
print ("PERP", np.exp(train_loss_history_RMSprop))
print ("_" * 50)



neural_n_gram_model_SGD, train_loss_history_SGD, val_loss_history_SGD = train_model_with (neural_n_gram_model_SGD,
                                                                      "SGD",
                                                                      SGD_LEARNING_RATE,
                                                                      N_EPOCHS,
                                                                      train_set_400,
                                                                      val_set_400,
                                                                      RMS_PROP_RHO,
                                                                      LEARNING_RATE_MULTIPLIER_PER_EPOCH
                                                                      )
    



input_sequence_SGD = [78]
input_sequence_SGD = neural_n_gram_model_SGD.generate_new_tokens(input_sequence_SGD, 200)
decoded_input_sequence_SGD = token_translator.decode_list(input_sequence_SGD)
print ("SGD text")
SGD_text = "".join(decoded_input_sequence_SGD).replace("</w>", " ")
print (SGD_text)
print (train_loss_history_SGD)
print ("_" * 50)




RMS_test_perp = np.exp(evaluate_model_on (neural_n_gram_model_RMSprop, test_set_400))
SGD_test_perp = np.exp(evaluate_model_on (neural_n_gram_model_SGD, test_set_400))
print (f"RMS test score = {RMS_test_perp}")
print (f"SGD test score = {SGD_test_perp}")


filename = "neural_bigram_test_set_perplexities.txt"

log_message (f"BPE k = {K}", filename)
log_message (f"n_epochs = {N_EPOCHS}", filename)
log_message ("_" * 100, filename)

log_message (f"RMSpror initial learning rate = {RMS_PROP_INITIAL_LR}", filename)
log_message (f"RMSprop rho = {RMS_PROP_RHO}", filename)

log_message(f"RMS test score = {RMS_test_perp}", filename)
log_message("_" * 50, filename)
log_message(f"RMS text = {RMS_text}", filename)
log_message("_"*75, filename)




log_message (f"SGD initial learning rate = {SGD_LEARNING_RATE}", filename)
log_message (f"SDG learning rate decay per epoch = {LEARNING_RATE_MULTIPLIER_PER_EPOCH}", filename)

log_message(f"SGD test score = {SGD_test_perp}", filename)
log_message("_" * 50, filename)
log_message(f"SGD text = {SGD_text}", filename)
log_message("_"*75, filename)

plot_perplexity (np.exp(train_loss_history_RMSprop), np.exp(val_loss_history_RMSprop), np.exp(train_loss_history_SGD), np.exp(val_loss_history_SGD))