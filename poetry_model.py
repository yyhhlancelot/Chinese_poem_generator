# *-* coding : utf-8 *-*
'''
	we build two kind of model here, one is with attention layer
	and another without, you can choose one to train the data.
	more details in the front_end.ipynb
	
	editor : yyh
	date : 2019-12-09
'''
import keras
from preprocess import *
import re
import os
from keras.callbacks import LambdaCallback
from keras.models import Input, Model, load_model
from keras.layers import LSTM, Dropout, Dense
from keras.optimizers import Adam
from keras.engine.topology import Layer

import numpy as np
import random

class PoetryModel(object):
    def __init__(self, config, atten = None, use_atten = False):
        '''
			class initialization
		'''
        self.model = None
        self.do_train = True
        self.config = config
        self.load_model = True
        self.use_atten = use_atten
        self.atten = atten
        
        # file preprocess
        self.word2id, self.id2word, self.word2idF, self.words, self.files_content = preprocess_file(config)
        self.poems = re.split(r"[]（）__]", self.files_content) #poem list
        self.poems = clean_data_form(self.config.form, self.poems)
        self.poems_num = len(self.poems)
        
        self.num_epochs = len(self.files_content) - (self.config.max_len + 1) * self.poems_num
        self.num_epochs /= self.config.batch_size
        self.num_epochs = int(self.num_epochs / 1.5)
        
        #load or train
        if os.path.exists(self.config.weight_file) and self.load_model:
            if self.use_atten:
                if self.atten:
                    self.model = load_model(self.config.weight_file, custom_objects = {'Attention' : self.atten})
                    print('model with Attention loaded')
                else:
                    # if forget to init attention
                    atten_temp = Attention(self.config.max_len)
                    self.model = load_model(self.config.weight_file, custom_objects = {'Attention' : atten_temp})
                    print('model with Attention loaded')
            else:
                self.model = load_model(self.config.weight_file)
                print('model without Attention loaded')
        else:
            self.train()
        
    def build_model(self):
        '''
			build model without attention
		'''
        print('build model')
        
        input_ = Input(shape = (self.config.max_len, len(self.words))) # one-hot
        lstm1_ = LSTM(output_dim = 512, activation = 'tanh', return_sequences = True)(input_)
        dropout1_ = Dropout(0.6)(lstm1_)
        lstm2_ = LSTM(output_dim = 256, activation = 'tanh')(dropout1_)
        dropout2_ = Dropout(0.6)(lstm2_)
        dense_ = Dense(len(self.words), activation = 'softmax')(dropout2_)
        self.model = Model(inputs = input_, output = dense_)
        optimizer_ = Adam(lr = self.config.learning_rate)
        self.model.compile(loss = 'categorical_crossentropy', optimizer = optimizer_, metrics = ['accuracy'])
        self.model.summary()
    
    def build_atten_model(self):
        '''
			add attention layer into the original model
		'''
        print('build model')
        
        input_ = Input(shape = (self.config.max_len, len(self.words)))
        lstm1_ = LSTM(output_dim = 512, activation = 'tanh', return_sequences = True)(input_)
        dropout1_ = Dropout(0.6)(lstm1_)
        lstm2_ = LSTM(output_dim =  256, activation = 'tanh', return_sequences = True)(dropout1_)
        atten_ = self.atten(lstm2_)
        dropout2_ = Dropout(0.6)(atten_)
        dense_ = Dense(len(self.words), activation = 'softmax')(dropout2_)
        self.model = Model(inputs = input_, output = dense_)
        optimizer_ = Adam(lr = self.config.learning_rate)
        self.model.compile(loss = 'categorical_crossentropy', optimizer = optimizer_, metrics = ['accuracy'])
        self.model.summary()
        
    def data_generator(self):
        '''
			data generator
		'''
        i = 0
        while 1:
            x = self.files_content[i : i + self.config.max_len]
            y = self.files_content[i + self.config.max_len]
            
            if ']' in x or ']' in y:
                i += 1
                continue
            y_vec = np.zeros(shape = (1, len(self.words)), dtype = np.bool)
            y_vec[0, self.word2idF(y)] = 1
            x_vec = np.zeros(shape = (1, self.config.max_len, len(self.words)), dtype = np.bool)
            for t, char in enumerate(x):
                x_vec[0, t, self.word2idF(char)] = 1
            yield x_vec, y_vec
            i += 1
            
    def predict_random(self, temperature = 1):
        '''
			internal test methods : randomly choose a poem with a first sentence, generate the next
		'''
        if not self.model:
            print('model not loaded')
            return
        index = random.randint(0, self.poems_num - 1)
        sentence = self.poems[index][: self.config.max_len]
        print('the first line : ', sentence)
        generate = self.predict_sentence(sentence, temperature = temperature)
        return generate
    
    def generate_sample_result(self, epoch, logs):
        '''
			write log and observe result with different diversity/temperature
		'''
        if epoch % 4 != 0:
            return
        with open('D:/code/project-research/Poems_generator/out_test/out.txt', 'a', encoding = 'utf-8') as f:
            f.write('==============epoch {}=============\n'.format(epoch))
        print('\n==============epoch {}=============\n'.format(epoch))
        
        # randomly choose sentence to test
        index = random.randint(0, self.poems_num - 1)
        sentence = self.poems[index][: self.config.max_len]
        print('the first line : ', sentence)
        
        for diversity in [0.7, 1, 1.4]:
            generate = self.predict_sentence(sentence, temperature = diversity)
            print('==========diversity {}============='.format(diversity))
            print(generate)
            with open('D:/code/project-research/Poems_generator/out_test/out.txt', 'a', encoding = 'utf-8') as f:
                f.write('{}\n'.format(generate))
    
    def predict_sentence(self, sentence, temperature = 1):
        '''
			according to the first max_len sentence, default generate the next 3 sentences
		'''
		
        if not self.model:
            return
        max_len = self.config.max_len
        if len(sentence) < max_len:
            print('length should not be less than ', max_len)
            return
        sentence = sentence[-max_len:]  # to get the last max_len words
        #print('the first line: ', sentence)
        generate = str(sentence)
        generate += self._preds(sentence, temperature, length = 24 - max_len)    
        return generate
    
    def sampling(self, preds, temperature = 1):
        '''
			use vector of propability to choose word,
			temperature means inclusiveness, while the value bigger, the result getting opener.
			when temperature = 0.5, which means the propability getting nearer, which made the result more conservative,
			temperature = 1, hold on and not change the value,
			temperature = 2, increase the interval of the probability, which made the result more open
		'''
		
        preds = np.asarray(preds).astype('float64')
        exp_preds = np.power(preds, temperature)
        soft_preds = exp_preds / np.sum(exp_preds)
        result_index_arr = np.random.choice(len(preds), size = 1, replace = False, p = soft_preds)
        return int(result_index_arr.squeeze())
        
    def _preds(self, sentence, temperature = 1, length = 23):
        '''
			input a sentence, and generate the next to get the whole chinese poem
		'''
		
        sentence = sentence[:self.config.max_len]
        generate = ''
        for i in range(length):
            # update prediction
            pred_char = self._pred(sentence, temperature)
            generate += pred_char 
            sentence = sentence[1:] + pred_char
        return generate
        
    def _pred(self, sentence, temperature = 1):
        '''according to a series of input/words, output a word'''
        if len(sentence) < self.config.max_len:
            print('in def of _pred, length error')
            return
        sentence = sentence[-self.config.max_len:] # to get the last max_len words
        x_test = np.zeros((1, self.config.max_len, len(self.words)))
        for t, char in enumerate(sentence):
            x_test[0, t, self.word2idF(char)] = 1
        preds = np.squeeze(self.model.predict(x_test))
        next_index = self.sampling(preds, temperature = temperature)
        next_char = self.id2word[next_index]
        return next_char
    
    def train(self):
        print('training')
        print('epochs = ', self.num_epochs)
        print('poems num = ', self.poems_num)
        print('total words length = ', len(self.files_content))
        if not self.model:
            if self.atten is None:
                print('start training without Attention')
                self.build_model()
            else:
                print('start training with Attention')
                self.build_atten_model()
        
        self.model.fit_generator(
        generator = self.data_generator(),
        verbose = True,
        steps_per_epoch = self.config.batch_size,
        epochs = self.num_epochs,
        callbacks = [
            keras.callbacks.ModelCheckpoint(self.config.weight_file, save_weights_only=False),
            LambdaCallback(on_epoch_end = self.generate_sample_result)
        ]
        )