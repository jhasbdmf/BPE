import re
from collections import Counter

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


def count_token_frequencies_of(unique_tokens: list):
    return Counter(unique_tokens)

#space-based tokenization
def tokenize_naively(corpus: str):
    return re.findall(r"[A-Za-z]+", corpus)


def get_string_chars(string: str):
    return re.findall(r"[A-Za-z]", string)