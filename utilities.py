import re
from collections import Counter
import os

def read_file_from (corpus_segment: str, file_subfolder: str, k: int = 2000):

    if file_subfolder.lower() == "corpus": 
        file_path = f"Corpus/Shakespeare_clean_{corpus_segment}.txt"
    elif file_subfolder.lower() == "learned_vocabularies":
        file_path = f"Learned_vocabularies/bpe_vocabulary_of_{corpus_segment}_set_with_k_{k}.txt"
    else:
        file_path = f"Tokenized_corpus/tokenized_{corpus_segment}_set_with_k_{k}.txt"
    
    try:
        with open(file_path , "r") as input_file:
            raw_text = input_file.read()
    except FileNotFoundError:
        parent_direcory = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
        alternative_file_path = os.path.join(parent_direcory, file_path)
        with open(alternative_file_path, "r") as input_file:
            raw_text = input_file.read()

    if file_subfolder.lower() == "corpus": 
        return raw_text
    else:
        tokenized_text = raw_text.split("\n")
        if tokenized_text[-1] == "":
            tokenized_text.pop()
    
        return tokenized_text
   

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