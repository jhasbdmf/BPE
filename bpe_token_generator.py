
import re
from collections import Counter
import time



def get_string_chars(string: str):
    return re.findall(r"[A-Za-z]", string)

#space-based tokenization
def tokenize_naively(corpus: str):
    return re.findall(r"[A-Za-z]+", corpus)

def count_token_frequencies_of(unique_tokens: list):
    return Counter(unique_tokens)

def get_charred_word_type_corpus_representation (corpus: str):

    #this one tells how frequent each word type is in the corpus
    non_tokenized_word_type_frequencies = count_token_frequencies_of(tokenize_naively(corpus.lower()))

    #this one also tells how frequent each word type is in the corpus
    #the difference is that here each word type is stored as
    #an array of chars at this stage of the program
    #later those chars will be merged into subword tokens
    tokenized_word_type_frequencies = {}

    for i in non_tokenized_word_type_frequencies:
        tokens_of_a_word_type = get_string_chars(i)
        tokens_of_a_word_type.append("</w>")
        tokenized_word_type_frequencies[tuple(tokens_of_a_word_type)] = non_tokenized_word_type_frequencies[i]
    
    return tokenized_word_type_frequencies



def write_tokens(tokens: list, n_iter: int, token_gen_duration: float):

    file_name = "Generated_tokens/bpe_tokens with k = " + str(n_iter) + ".txt"
    with open(file_name, "w") as output_file:
        #output_file.write(f"Generation of {n_iter} tokens took {token_gen_duration:.4f} seconds")
        for token in tokens:
            output_file.write(f"{token}\n") 

def bpe (vocab: list, corpus_representation: dict, n_iter: int):
    start = time.time()
    for current_iter in range(n_iter):

 
        inter_word_counter = {}
        for word_type, word_type_frequency in corpus_representation.items():

            #how often a token pair occurs in a word type
            intra_word_counter = Counter(zip(word_type, word_type[1:]))
            
            for key in intra_word_counter:

                #how often a token occurs in a word type * word type frequency in a corpus
                intermediate_key_frequency = intra_word_counter[key] * word_type_frequency

                #add intermediate key frequency across word types
                #in which in occurs
          
                if not key in inter_word_counter:
                    inter_word_counter[key] = intermediate_key_frequency
                else:
                    inter_word_counter[key] += intermediate_key_frequency
       

        new_token_tuple = max(inter_word_counter, key=inter_word_counter.get)
        new_token_element1 = new_token_tuple[0]
        new_token_element2 = new_token_tuple[1]
        new_token = new_token_element1 + new_token_element2
        print ("new token ", new_token)
        if not new_token in vocab:
            vocab.append(new_token)
        else:
            print ("!!!NON-UNIQUE TOKEN GENERATION ATTEMPT\n")

        #replace pairs of new_token_element1, new_token_element2
        #with new_token_element1 + new_token_element2
        #in all word types of a corpus
        new_corpus_representation = {}
        for word_type, word_type_frequency in corpus_representation.items():
       
            new_word_type = []
            i = 0
            while i < len(word_type):
                if word_type[i] == new_token_element1 and word_type[i+1] == new_token_element2:
                    new_word_type.append(new_token)
                    i+=2
                else:
                    new_word_type.append(word_type[i])
                    i += 1
      
            new_corpus_representation[tuple(new_word_type)] = word_type_frequency 
     
        corpus_representation = new_corpus_representation
        if (current_iter+1) >= 1000 and (current_iter+1) % 200 == 0:
            end = time.time()
            elapsed = end - start
            write_tokens(vocab, current_iter+1, elapsed)
    
    return vocab





with open("Corpus/Shakespeare_clean_train.txt", "r") as input_file:
    raw_text = input_file.read()



corpus_chars = get_string_chars(raw_text.lower())
vocabulary = list(set(corpus_chars))
vocabulary.append("</w>")


tokenized_word_type_frequencies = get_charred_word_type_corpus_representation(raw_text)


start = time.time()
print(bpe(vocabulary, tokenized_word_type_frequencies, 2000))
end = time.time()
elapsed = end - start
print(f"Elapsed time: {elapsed:.4f} seconds")







  