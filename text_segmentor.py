import re
from utilities import tokenize_naively, get_string_chars


def get_word_type_counts_with_positions_in_a_corpus (corpus: str):
    corpus_tokens = tokenize_naively(corpus.lower())


    corpus_representation: dict[tuple, list] = {} 
    for word_token_index, word_token in enumerate(corpus_tokens):

        
        word_token_charred = get_string_chars(word_token)
        
        word_token_charred.append("</w>")
        print (word_token_charred )
        print (type(word_token_charred) )
        word_token_chars = tuple(word_token_charred)
        
        if not word_token_chars in corpus_representation:
            corpus_representation[word_token_chars] = [word_token_index]
        else:
            corpus_representation[word_token_chars].append(word_token_index)
    return corpus_representation


def tokenize_corpus (corpus_representation: dict, vocab: list):

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


#read vocab from a .txt file

with open("Generated_tokens/bpe_tokens with k = 2000.txt", "r") as vocabulary_file:
    vocabulary = vocabulary_file.read()

#remove empty string from end of vocab should it have been read into it
vocabulary = vocabulary.split("\n")
if vocabulary[len(vocabulary)-1] == "":
    vocabulary.pop()

with open("Corpus/Shakespeare_clean_valid.txt", "r") as validation_file:
    corpus = validation_file.read()
#corpus = corpus.replace(" ", "</w>")
#print (corpus[:100])

charred_word_types_with_frequencies_and_positions = get_word_type_counts_with_positions_in_a_corpus (corpus)

#for i in charred_word_types_with_frequencies_and_positions:
#    print (i, charred_word_types_with_frequencies_and_positions[i])

print (tokenize_corpus(charred_word_types_with_frequencies_and_positions, vocabulary))





