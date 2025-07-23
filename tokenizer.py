import nltk
import re
from collections import Counter

def extract_letters(string: str):
    return " ".join(re.findall(r"[A-Za-z]+", string))

def tokenize_naively(corpus: str):
    return re.findall(r"[A-Za-z]+", corpus)

def count_token_frequencies_of(unique_tokens: list):
    return Counter(unique_tokens)

def get_unique_chars (string: str):
    unique_chars = set()
    for char in string:
        unique_chars.add(char)
    
    return list(unique_chars)

def get_all_vocab_combinations (vocab: list):

    
    vocab_combinations = []
    for i in range(len(vocab)):
        for j in range (i+1, len(vocab)):
            vocab_combination = vocab[i] + vocab[j]
           
            if not vocab_combination in vocab:
                vocab_combinations.append(vocab_combination)

    return vocab_combinations 

def count_token_frequency_in (token: str, word_frequencies: Counter):
    
    token_count = 0

    for word, word_frequency in word_frequencies.items():
        token_count += word.count(token) * word_frequency

    return token_count


def get_all_vocab_item_frequencies_in (vocab_cand: list, word_frequencies: str, all_token_frequencies: dict):
    candidate_token_frequencies = {}

    for candidate in vocab_cand:
        if candidate not in all_token_frequencies.keys():
            #all_token_frequencies[candidate] = corpus.count(candidate)
            all_token_frequencies[candidate] = count_token_frequency_in (candidate, word_frequencies)
        candidate_token_frequencies[candidate] = all_token_frequencies[candidate]
    
    return candidate_token_frequencies

def get_new_vocab_item_from (vocab_candidates: list, word_frequencies: Counter, token_frequencies: dict):


    candidate_token_frequencies = get_all_vocab_item_frequencies_in (vocab_candidates, word_frequencies, token_frequencies)
    best_vocab_candidate = max(candidate_token_frequencies, key=candidate_token_frequencies.get)
    return best_vocab_candidate
 


def bpe(vocab: list, word_frequencies: Counter, n_iter: int):

    new_vocab = vocab.copy()

    token_freq_count = {}
    for i in range(n_iter):
        
        vocab_pairs = get_all_vocab_combinations(new_vocab)
        new_vocab_item = get_new_vocab_item_from (vocab_pairs, word_frequencies, token_freq_count)
        print ("New token:\n", new_vocab_item)
        new_vocab.append(new_vocab_item)
        #print (new_vocab)

    return new_vocab
   




with open("shakespeare.txt", "r") as f:
    raw_text = f.read()

print ("Raw text length in chars:\n", len(raw_text), "\n")
print ("First raw text chars:\n", raw_text[:9], "\n")

clean_text = extract_letters(raw_text.lower())
print ("First chars of cleaned text:\n", clean_text[:2000], "\n")

initial_vocabulary = get_unique_chars(clean_text)
print ("Initial vocabulary:\n", initial_vocabulary, "\n")

print (len(get_all_vocab_combinations (initial_vocabulary)))

word_frequency_distribution = count_token_frequencies_of(tokenize_naively(raw_text.lower()))

print (word_frequency_distribution)
print (len(word_frequency_distribution))
print (type(word_frequency_distribution))

print ("bpe result:\n", bpe(initial_vocabulary, word_frequency_distribution, 100))


