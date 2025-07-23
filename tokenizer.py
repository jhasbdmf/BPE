#import nltk
import re
from collections import Counter

def extract_letters(string: str):
    return " ".join(re.findall(r"[A-Za-z]+", string))

def get_corpus_chars(corpus: str):
    return " ".join(re.findall(r"[A-Za-z]+", corpus))

def get_unique_corpus_chars (corpus: str):
    unique_chars = set(corpus)

    
    return list(unique_chars)

def tokenize_naively(corpus: str):
    return re.findall(r"[A-Za-z]+", corpus)

def count_token_frequencies_of(unique_tokens: list):
    return Counter(unique_tokens)


 
def bpe(corpus_tokens: list, n_iter: int):

    #get unique corpus chars
    vocab = set(corpus_tokens)


    for _ in range (n_iter):

        #count frequencies of all pairs of adjacent corpus tokens
        token_frequencies = Counter(zip(corpus_tokens, corpus_tokens[1:]))

        #print (token_frequencies )

        #add most frequent adjacent token pair into dictionary
        #new_token_element1, new_token_element2, freq = token_frequencies.most_common(1)[0]
        most_frequent_token_combination, _ = token_frequencies.most_common(1)[0]
        new_token_element1 = most_frequent_token_combination[0]
        new_token_element2 = most_frequent_token_combination[1]
        new_token = new_token_element1 + new_token_element2
        #print (most_frequent_token_combination)
        print ("New token:", new_token)
        vocab.add(new_token)

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
    return vocab



with open("shakespeare.txt", "r") as f:
    raw_text = f.read()

print ("Raw text length in chars:\n", len(raw_text), "\n")
print ("First raw text chars:\n", raw_text[:9], "\n")

clean_text = extract_letters(raw_text.lower())
print ("First chars of cleaned text:\n", clean_text[:100], "\n")

initial_vocabulary = get_unique_corpus_chars(clean_text)
print ("Initial vocabulary:\n", initial_vocabulary, "\n")







corpus_chars = get_corpus_chars(raw_text.lower())
print ("Corpus chars:\n", corpus_chars[:100])

print ("bpe generated vocab:\n", bpe(corpus_chars, 1000))
