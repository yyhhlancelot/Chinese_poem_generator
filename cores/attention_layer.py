# *-* coding : utf-8 *-*
'''
	additional Layer : Attention
	editor : yyh
	date : 2019-12-09
'''
from keras.engine.topology import Layer
from keras import backend as K
import numpy as np
from keras import constraints, initializers, regularizers
class Attention(Layer):
    def __init__(self, step_dim = 6, W_regularizer = None, b_regularizer = None, W_constraints = None, b_constraints = None, **kwargs):
        self.support_masking = True
        self.initializer = initializers.get('glorot_uniform')
        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)
        self.W_constraints = constraints.get(W_constraints)
        self.b_constraints = constraints.get(b_constraints)
        
        self.step_dim = step_dim
        self.bias = True
        self.features_dim = 0
        super(Attention, self).__init__(**kwargs)
        
    def build(self, input_shape):
        assert len(input_shape) == 3
        
        self.W = self.add_weight(name = '{}_W'.format(self.name),
                                shape = (input_shape[-1], ),
                                initializer = self.initializer,
                                regularizer = self.W_regularizer,
                                trainable = True
                                )
        self.features_dim = input_shape[-1]
        if self.bias:
            self.b = self.add_weight(name = '{}_b'.format(self.name),
                                    shape = (input_shape[1], ),
                                     initializer = self.initializer,
                                     regularizer = self.b_regularizer,
                                     trainable = True
                                    )
        else:
            self.b = None
        super(Attention, self).build(input_shape)
    
    def compute_mask(self, input, input_mask = None):
        return None
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.features_dim)
    
    def call(self, x, mask = None):
        '''write sort of logic of your self-defined layer here'''
        step_dim = self.step_dim
        features_dim = self.features_dim
        eij = K.reshape(K.dot(K.reshape(x, (-1, features_dim)), K.reshape(self.W, (features_dim, 1))), (-1, step_dim))
        if self.bias:
            eij += self.b
        eij = K.tanh(eij)
        
        alpha = K.exp(eij)
        alpha /= K.cast(K.sum(alpha, axis = 1, keepdims = True), K.floatx())
        
        if mask is not None:
            alpha = K.cast(K.dot(alpha, mask), K.floatx())
        
        alpha = K.expand_dims(alpha, axis = 2)
        weighted_input = K.cast(x * alpha, K.floatx())
        return K.sum(weighted_input, axis = 1) # means no return sequences, absolutely originly no
        
        