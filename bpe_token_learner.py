from collections import Counter
import time
from utilities import get_charred_word_type_corpus_representation, get_string_chars, read_file_from
import os

def write_tokens(tokens: list, n_iter: int, corpus_segment: str, token_gen_duration: float):

    # 1) figure out parent directory
    cwd       = os.getcwd()
    #parent    = os.path.abspath(os.path.join(cwd, os.pardir))

    # 2) build (and create) your output folder under the parent
    #out_dir   = os.path.join(parent, "Learned_vocabularies")

    out_dir = os.path.join(cwd, "Learned_vocabularies")
    os.makedirs(out_dir, exist_ok=True)

    # 3) build your file name and full path
    file_name = f"bpe_vocabulary_of_{corpus_segment}_set_with_k_{n_iter}.txt"
    file_path = os.path.join(out_dir, file_name)

    # 4) write out your tokens
    with open(file_path, "w", encoding="utf-8") as out_f:
        for token in tokens:
            out_f.write(f"{token}\n")
    print(f"Wrote {len(tokens)} tokens to {file_path}")


def bpe (vocab: list, corpus_representation: dict, n_iter: int, corpus_segment: str):
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
                    i += 2
                else:
                    new_word_type.append(word_type[i])
                    i += 1
      
            new_corpus_representation[tuple(new_word_type)] = word_type_frequency 
     
        corpus_representation = new_corpus_representation
        if (current_iter+1) >= 1000 and (current_iter+1) % 200 == 0:
            end = time.time()
            elapsed = end - start
            write_tokens(vocab, current_iter+1, corpus_segment, elapsed)
    
    return vocab


CORPUS_SEGMENT = "train"


raw_text = read_file_from(CORPUS_SEGMENT, "corpus")




corpus_chars = get_string_chars(raw_text.lower())
vocabulary = list(set(corpus_chars))
vocabulary.append("</w>")


tokenized_word_type_frequencies = get_charred_word_type_corpus_representation(raw_text)


start = time.time()
print(bpe(vocabulary, tokenized_word_type_frequencies, 2000, CORPUS_SEGMENT))
end = time.time()
elapsed = end - start
print(f"Elapsed time: {elapsed:.4f} seconds")







  