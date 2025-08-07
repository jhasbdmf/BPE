import re
from utilities import tokenize_naively, get_string_chars, read_file_from
import os

def write_tokenized_corpus_segment (corpus_chunk_name: str, segmented_corpus_chunk: list):
    file_path = f"Tokenized_corpus/Tokenized_{corpus_chunk_name}_set.txt"
    try:
        with open(file_path, "w") as output_file:
            for token in segmented_corpus:
                output_file.write(f"{token}\n")
    except FileNotFoundError:
        parent_direcory = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
        alternative_file_path = os.path.join(parent_direcory, file_path)
        with open(alternative_file_path, "w") as output_file:
            for token in segmented_corpus_chunk:
                output_file.write(f"{token}\n")


def get_word_type_counts_with_positions_in_a_corpus (corpus: str):
    corpus_tokens = tokenize_naively(corpus.lower())


    corpus_representation: dict[tuple, list] = {} 
    for word_token_index, word_token in enumerate(corpus_tokens):

        
        word_token_charred = get_string_chars(word_token)
        
        word_token_charred.append("</w>")
        #print (word_token_charred)
        #print (type(word_token_charred) )
        word_token_chars = tuple(word_token_charred)
        
        if not word_token_chars in corpus_representation:
            corpus_representation[word_token_chars] = [word_token_index]
        else:
            corpus_representation[word_token_chars].append(word_token_index)
    return corpus_representation, len(corpus_tokens)



def merge_corpus_chars (corpus_representation: dict, vocab: list):

    for vocab_item in vocab:
    

        #skip over alphabet chars
        if len(vocab_item) > 1 and vocab_item != "</w>":

            new_corpus_representation: dict [tuple, list] = {}
            
            for word_type, word_type_positions in corpus_representation.items():
                new_word_type = []
                i = 0
                while i < len (word_type):
                    if (i< len (word_type)-1) and (word_type[i]+word_type[i+1] == vocab_item):
                        new_word_type.append(vocab_item)
                        i += 2
                    else:
                        new_word_type.append(word_type[i])
                        i += 1
                new_corpus_representation [tuple(new_word_type)] = word_type_positions 
            corpus_representation = new_corpus_representation    
    return corpus_representation

def reconstruct_corpus_from (corpus_representation: dict, n_words_in_corpus: int):

    reconstructed_corpus = [0] * n_words_in_corpus

    for word_type, word_type_positions_in_corpus in corpus_representation.items():
        for word_type_position in word_type_positions_in_corpus:
            reconstructed_corpus[word_type_position] = word_type 

    #return reconstructed_corpus
    return [x for tup in reconstructed_corpus for x in tup]

def tokenize_sequence(corpus: str, vocabulary: list):
    charred_word_types_with_frequencies_and_positions, number_of_words_in_a_corpus = get_word_type_counts_with_positions_in_a_corpus (corpus)
    corpus_tokens_compressed = merge_corpus_chars(charred_word_types_with_frequencies_and_positions, vocabulary)
    reconstructed_corpus = reconstruct_corpus_from(corpus_tokens_compressed, number_of_words_in_a_corpus)
    return reconstructed_corpus



CORPUS_SEGMENT = "test"
raw_corpus = read_file_from(CORPUS_SEGMENT, "corpus")
vocabulary = read_file_from("train", "Learned_vocabularies")


segmented_corpus = tokenize_sequence (raw_corpus, vocabulary)
write_tokenized_corpus_segment (CORPUS_SEGMENT, segmented_corpus)

