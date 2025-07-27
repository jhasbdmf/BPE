from collections import Counter
from text_segmentor import tokenize_sequence


def get_n_gram_counts_from (n: int, corpus_tokens: list):
    n_gram_list = []
   
    for token_index, token in enumerate(corpus_tokens):
        if (token_index + n - 1) < len (corpus_tokens):
            n_gram = [token]
            for j in range(1, n):
                n_gram.append(corpus_tokens[token_index + j])
            n_gram_list.append(tuple(n_gram))

    return Counter(n_gram_list)

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

def get_next_most_probable_token_after(sequence: list, sorted_n_grams: list, n_gram_counts: list):

    left_index = get_leftmost_index_of_in(sequence, sorted_n_grams)
    right_index = get_rightmost_index_of_in(sequence, sorted_n_grams)

    if left_index > -1 and right_index > -1 and right_index >= left_index:
        #print (sorted_n_grams[left_index:(right_index+1)])
        max_frequency, n_gram_with_max_frequency = 0, ()
        for most_frequent_n_gram_candidate in sorted_n_grams[left_index:(right_index+1)]:
            #print (most_frequent_n_gram_candidate, n_gram_counts[most_frequent_n_gram_candidate])
            if n_gram_counts[most_frequent_n_gram_candidate] > max_frequency:
                max_frequency = n_gram_counts[most_frequent_n_gram_candidate]
                n_gram_with_max_frequency = most_frequent_n_gram_candidate
        #next_token = n_gram_with_max_frequency[len(n_gram_with_max_frequency)-1].replace("</w>", " ")
        next_token = n_gram_with_max_frequency[len(n_gram_with_max_frequency)-1]
    else:
        next_token = "and "
    return next_token


with open("Tokenized corpus.txt", "r") as tokenized_corpus_file:
    corpus_tokens = tokenized_corpus_file.read()

#remove empty string from end of vocab should it have been read into it
corpus_tokens = corpus_tokens.split("\n")
if corpus_tokens[len(corpus_tokens)-1] == "":
    corpus_tokens.pop()


#print (corpus_tokens)
#print (type(corpus_tokens))
MAX_N_GRAM_LENGTH = 4
counts_of_4_grams = get_n_gram_counts_from(MAX_N_GRAM_LENGTH, corpus_tokens)



#for i in n_gram_counts.most_common(100):
#    print (i)
#for n_gram in n_gram_counts:
#    n_gram_counts[n_gram] /= len(n_gram_list)

sorted_4_grams = sorted(counts_of_4_grams)

#with open("n_grams.txt", "w") as output_file:
#    for i in sorted_n_grams:
#        output_file.write(str(i) + "\n")


#for n_gram in n_gram_counts[:1000]:
#    print (n_gram, f"{n_gram_counts[n_gram]:.7f}")
#print (len(n_gram_counts))
#print (len(n_gram_list))

while True:
    input_string = input("Give me three words separated by single spaces to autocomplete: ")
    #input_string = input_string.strip()
    for _ in range(10):
        input_string_tail = tuple(tokenize_sequence(input_string)[-(MAX_N_GRAM_LENGTH-1):])

        print ("input ngram: ", input_string_tail)

        #left_index = get_leftmost_index_of_in(input_string_tail, sorted_n_grams)
        #right_index = get_rightmost_index_of_in(input_string_tail, sorted_n_grams)
        #if left_index > -1 and right_index > -1 and right_index >= left_index:
        #    print (sorted_n_grams[left_index:(right_index+1)])
        ##    max_frequency, n_gram_with_max_frequency = 0, ()
        #    for most_frequent_n_gram_candidate in sorted_n_grams[left_index:(right_index+1)]:
        #        print (most_frequent_n_gram_candidate, n_gram_counts[most_frequent_n_gram_candidate])
        #        if n_gram_counts[most_frequent_n_gram_candidate] > max_frequency:
        #            max_frequency = n_gram_counts[most_frequent_n_gram_candidate]
        #            n_gram_with_max_frequency = most_frequent_n_gram_candidate
        #    next_token = n_gram_with_max_frequency[len(n_gram_with_max_frequency)-1].replace("</w>", " ")
        #else:
        #    next_token = "and "
        next_token = get_next_most_probable_token_after(input_string_tail, sorted_4_grams, counts_of_4_grams)
        input_string += next_token
        print ("Next token is: ", next_token)
        print ("Input sequence now is: ", input_string)
    
