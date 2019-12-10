# *-* coding : utf-8 *-*
'''
	preprocess data
	editor : yyh
	date : 2019-12-09
'''
import re

puncs = [']', '[', '(', ')', '{', '}', ':', '《', '》']
def preprocess_file(config):
    '''
		text preprocess, generate a file content(str)
	'''
    files_content = ''
    with open(config.poetry_file, 'r', encoding = 'utf-8') as f:
        for line in f:
            # add "]" means the end of a poem
            line = line.split(':')[1]
            if len(line) < config.form:
                continue
            for char in puncs:
                line = line.replace(char, "")
            if line[config.form] == "，":
                files_content += line.strip() + "]"
    
    words = sorted(list(files_content)) # all word list, include repeat
    
    stopwords = [']']
    for word in list(words):
        if word in stopwords:
            words.remove(word)
    
    counted_words = {}
    for word in words:
        if word in counted_words:
            counted_words[word] += 1
        else:
            counted_words[word] = 1
    
    # erase low occ
    erase = []
    for key in counted_words:
        if counted_words[key] <= 2:
            erase.append(key)
    for key in erase:
        del counted_words[key]
    wordPairs = sorted(counted_words.items(), key = lambda x : -x[1])
    
    words, counts = zip(*wordPairs) # dict, not include repeat
    
    # ->
    word2id =  dict((c, i) for i, c in enumerate(words))
    id2word = dict((i, c) for i, c in enumerate(words))
    word2idF = lambda x : word2id.get(x, 0)
    return word2id, id2word, word2idF, words, files_content

def clean_data_form(form, poems):
    '''
		accordding to the characters num to choose the training data, or not choose
	'''
    if form == 5 or form == 7:
        for poem in list(poems):
            if len(poem) >= form + 1 and poem[form] != '，' or len(poem) < 4:
                poems.remove(poem)   
        return poems
    else:
        return poems