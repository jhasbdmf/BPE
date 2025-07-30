from collections import Counter
import math
from text_segmentor import tokenize_sequence



#llm rewrote my method which generated n_grams with a nested loop
def get_n_gram_counts_from(n: int, corpus_tokens: list):
    # Build n sliding iterators over the token list
    n_gram_iterators = [corpus_tokens[i:] for i in range(n)]
    # Zip to produce n-gram tuples from these slices
    n_grams = zip(*n_gram_iterators)
    return Counter(n_grams)


def get_frequent_n_grams (n_gram_counts: Counter, frequency_threshold: int):
    return {key: value for key, value in n_gram_counts.items() if value > frequency_threshold}


def get_n_grams_infos_from (n: int, corpus_tokens: list):
    counts_of_n_grams = get_n_gram_counts_from(n, corpus_tokens)
    #remove singleton n_grams from n_gram list
    #counts_of_n_grams = get_frequent_n_grams(counts_of_n_grams, 1)
    sorted_n_grams = sorted(counts_of_n_grams)
    return counts_of_n_grams, sorted_n_grams  


def get_leftmost_index_of_in (n_gram_prefix: tuple, n_gram_list: list):

    n = len(n_gram_prefix)
    left_bound = 0
    right_bound = len(n_gram_list) - 1

    result = -1
    
    while left_bound <= right_bound:
        
        mid = (right_bound + left_bound) // 2

        if n_gram_list[mid][:n] >= n_gram_prefix:
            result = mid
            right_bound = mid - 1
        else:
            left_bound = mid + 1

    return result


def get_rightmost_index_of_in (n_gram_prefix: tuple, n_gram_list: list):

    n = len(n_gram_prefix)
    left_bound = 0
    right_bound = len(n_gram_list) - 1
    result = -1
    
    while left_bound <= right_bound:
        mid = (right_bound + left_bound) // 2
      
        if n_gram_list[mid][:n] <= n_gram_prefix:
            left_bound = mid + 1
            result = mid
        else:
            right_bound = mid - 1

    return result


def get_next_most_probable_token_after_via_backoff(sequence: list, sorted_n_grams: list, n_gram_counts: list):

    next_token_found = False
    for i in range(0, len(sorted_n_grams)):

        left_index = get_leftmost_index_of_in(sequence[i:], sorted_n_grams[i])
        right_index = get_rightmost_index_of_in(sequence[i:], sorted_n_grams[i])

        if left_index > -1 and right_index > -1 and right_index >= left_index:

            #print (sorted_n_grams[left_index:(right_index+1)])

            max_frequency, n_gram_with_max_frequency = 0, ()
            for most_frequent_n_gram_candidate in sorted_n_grams[i][left_index:(right_index+1)]:

                #print (most_frequent_n_gram_candidate, n_gram_counts[most_frequent_n_gram_candidate])

                if n_gram_counts[i][most_frequent_n_gram_candidate] > max_frequency:
                    max_frequency = n_gram_counts[i][most_frequent_n_gram_candidate]
                    n_gram_with_max_frequency = most_frequent_n_gram_candidate
            
            next_token = n_gram_with_max_frequency[len(n_gram_with_max_frequency)-1]
            next_token_found = True
            break

    if not next_token_found:
        next_token = "or</w>"

    return next_token


def go_into_generation_mode ():
    while True:
        input_string = input("Give me 3 words separated by single spaces to autocomplete: ")
        input_string_tokens = tokenize_sequence(input_string)
        
        for _ in range(200):
            input_string_tail_tokens = tuple(input_string_tokens[-(MAX_N_GRAM_LENGTH-1):])
            next_token = get_next_most_probable_token_after_via_backoff(input_string_tail_tokens, sorted_n_grams, counts_of_n_grams)
            input_string_tokens.append(next_token)
        
        print ("Generated sequence now is: ", "".join(input_string_tokens).replace("</w>"," "))

def get_perplexity_score_of_via(corpus_segment: str, n_gram_counts: list, max_n_gram_len: int, backoff_discount_factor: float):

    with open(f"Tokenized {corpus_segment} set.txt", "r") as tokenized_corpus_file:
        corpus_tokens = tokenized_corpus_file.read()

    corpus_tokens = corpus_tokens.split("\n")
    if corpus_tokens[len(corpus_tokens)-1] == "":
        corpus_tokens.pop()

    total_number_of_unigrams = sum(n_gram_counts[-1].values())
    net_log_P = 0

    for i in range (max_n_gram_len - 1, len (corpus_tokens)-1):
        sequence = tuple(corpus_tokens[i- max_n_gram_len + 1 : i+1])
        n_gram_found = False
        for j in range(len(n_gram_counts)):
            
            sequence_freq = n_gram_counts[j][sequence[j:]]
            #if sequence[j:] in n_gram_counts[j]:
            if sequence_freq > 0:
                current_log_P = (backoff_discount_factor ** j) * sequence_freq 
                if len (sequence[j:-1]) > 0:
                    
                    perplexity_divisor = n_gram_counts[j+1][sequence[j:-1]]
                    
                else:
                    perplexity_divisor = total_number_of_unigrams
                current_log_P /= perplexity_divisor
                net_log_P += math.log(current_log_P) 

                n_gram_found = True
                break
        if not n_gram_found:
            print ("NO N_GRAM FOUND FOR: ", sequence[j:])
    len_corpus = len (corpus_tokens) - max_n_gram_len
    perplexity = math.exp(-1* net_log_P / len_corpus)
 
    return perplexity


with open("Tokenized train set.txt", "r") as tokenized_corpus_file:
    corpus_tokens = tokenized_corpus_file.read()

#remove empty string from end of vocab should it have been read into it
corpus_tokens = corpus_tokens.split("\n")
if corpus_tokens[len(corpus_tokens)-1] == "":
    corpus_tokens.pop()

MAX_N_GRAM_LENGTH = 5

counts_of_n_grams, sorted_n_grams = [], []
for i in range (MAX_N_GRAM_LENGTH, 0, -1):
    counts_of_i_grams, sorted_i_grams = get_n_grams_infos_from(i, corpus_tokens)
    counts_of_n_grams.append(counts_of_i_grams)
    sorted_n_grams.append(sorted_i_grams)

perplexity = get_perplexity_score_of_via ("valid", counts_of_n_grams, MAX_N_GRAM_LENGTH, 0.4)
print (f"For n = {MAX_N_GRAM_LENGTH} perplexity is equal to {perplexity}")

#go_into_generation_mode ()

