# *-* coding : utf-8 *-*
'''
	configuration
	editor : yyh
	date : 2019-12-09
'''

class Config(object):
    '''form_ equals 5 means we choose 5 character poems as our training data,
    equals 7 means we choose 7 character poems as training data,
    while other nums means we do not choose.
    max equals form plus 1'''
    def __init__(self, poetry_file, weight_file, max_len, batch_size, learning_rate, form):
        self.poetry_file = poetry_file
        self.weight_file = weight_file
        self.max_len = max_len
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.form = form