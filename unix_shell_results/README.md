This were the commands for the generated files:

all_words.txt

 tr -sc 'A-Za-z' '\n' < shakespeare.txt > unix_shell_results/all_words.txt 
 -> Only Strings with letters in the 'A-Za-z' range 
 -> No numbers or Strings that are =! the symbols in die range
 -> Number of words (used cmd: wc -l): 928012

unique_words.txt

tr -sc 'A-Za-z' '\n' < shakespeare.txt | sort | uniq -c | sort -nr > unix_shell_results/unique_words.txt
-> Unique Words sorted by frequency 
-> Number unique words (used cmd: wc -l): 29454

unique_words_ignore_case.txt

tr -sc 'A-Za-z' '\n' < shakespeare.txt | tr 'A-Z' 'a-z' | sort | uniq -c | sort -nr > unix_shell_results/unique_words_ignore_case.txt
-> Unique words by 
-> Number of unique words without case sensitiveness (used cmd: wc -l): 23683