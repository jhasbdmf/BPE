import nltk
from nltk.stem import WordNetLemmatizer
from utilities import read_file_from


lemmatizer = WordNetLemmatizer()

#with open("shakespeare.txt", "r", encoding="utf-8") as f:
#    text = f.read()
#print(text)

text = read_file_from("full", "corpus")

print (text[0:9])

text_words = text.lower().split()


print (text_words[0:9])

unique_words = set()
unique_words_count = dict()


for word in text_words:
    #unique_words.add(word)

    if not word.isnumeric():
        word_lemma = lemmatizer.lemmatize(word)
        unique_words.add(word_lemma)
        if not word_lemma in unique_words_count.keys():
            unique_words_count[word_lemma] = 1
        else:
            unique_words_count[word_lemma] += 1

print ("all words", len(text_words))
print ("unique words", len(unique_words))
#print (len(unique_words_count))
#print ("most frequent words: ", unique_words_count[0:9])

sorted_dict = dict(sorted(unique_words_count.items(), key=lambda item: item[1], reverse=True))

print("Most frequent items: ", list(sorted_dict.items())[:30])