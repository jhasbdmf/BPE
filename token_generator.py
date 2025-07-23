#import nltk
import re
from collections import Counter

def get_corpus_chars(corpus: str):
    return " ".join(re.findall(r"[A-Za-z]+", corpus))

def tokenize_naively(corpus: str):
    return re.findall(r"[A-Za-z]+", corpus)

def count_token_frequencies_of(unique_tokens: list):
    return Counter(unique_tokens)

def bpe(corpus_tokens: list, n_iter: int):

    #get unique corpus chars
    #!!vocab should be a list to preserve addind order
    vocab = list(set(corpus_tokens))


    for current_iter in range (n_iter):

        #count frequencies of all pairs of adjacent corpus tokens
        token_frequencies = Counter(zip(corpus_tokens, corpus_tokens[1:]))

        #print (token_frequencies )

        #add most frequent adjacent token pair into dictionary
        most_frequent_token_combination, _ = token_frequencies.most_common(1)[0]
        new_token_element1 = most_frequent_token_combination[0]
        new_token_element2 = most_frequent_token_combination[1]
        new_token = new_token_element1 + new_token_element2
   
        print ("New token:", new_token)
        if not new_token in vocab:
            vocab.append(new_token)
        else:
            print ("!!!NON-UNIQUE TOKEN GENERATION ATTEMPT\n")

        #replace tokenized corpus with another tokenized corpus
        #in which new_token_element1 and new_token_element2 are
        #replaced with concat(new_token_element1, new_token_element2)
        i=0
        new_corpus_tokens = []
        while i < len(corpus_tokens)-1:
            if corpus_tokens[i] == new_token_element1 and corpus_tokens[i+1] == new_token_element2:
                new_corpus_tokens.append(new_token)
                i += 2
            else:
                new_corpus_tokens.append(corpus_tokens[i])
                i +=1
        corpus_tokens = new_corpus_tokens

        if (current_iter+1)>= 1000 and (current_iter+1) % 200 == 0:
            write_tokens(vocab, current_iter+1)
    return vocab

def write_tokens(tokens: list, n_iter: int):

    file_name = "quasi_bpe_tokens with k = " + str(n_iter) + ".txt"
    with open(file_name, "w") as output_file:
        for token in tokens:
            output_file.write(f"{token}\n") 

with open("shakespeare.txt", "r") as input_file:
    raw_text = input_file.read()

print ("Raw text length in chars:\n", len(raw_text), "\n")
print ("First raw text chars:\n", raw_text[:9], "\n")

corpus_chars = get_corpus_chars(raw_text.lower())
print ("Corpus chars:\n", corpus_chars[:100])

#generated_tokens = bpe(corpus_chars, 2000)
#print ("bpe generated vocab:\n", generated_tokens)






  