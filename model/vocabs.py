import nengo
import nengo_spa as spa
import numpy as np

def key_range(start, stop, step=1):

	assert start < stop

	digits = []
	digits.append('D'+str(step)+'.unitary()')
	if step != start:
		digits.append('D'+str(start)+'.unitary()')
	for i in range(start+step, stop, step):
		digits.append('D'+str(i)+'='+digits[-1].split('=')[0].split('.')[0]+'*D'+str(step)) # D_n = D_{n-step}*D_{step}
	return digits

def create_vocab(D, keys, rng):
	vocab = spa.Vocabulary(int(D), pointer_gen=rng)
	if type(keys) is list:
		vocab.populate(';'.join(keys))
	else:
		vocab.populate(keys)
	return vocab

def create_vocabs(D, seed):
	vocabs = {}
	rng = np.random.RandomState(seed)

	compare_keys 					= 'MORE ; LESS'
	PRIM_keys 						= 'GET_V ; GET_COM ; GET_ADD ; SET_COM ; SET_ADD ; SET_M'
	even_digit_keys					= key_range(2,10,2)
	digit_keys 						= key_range(1,11,1) + ['D0=D1*~D1']
	instructions_keys				= [PRIM_keys] + digit_keys
	vision_keys 					= digit_keys + ['FIXATE'] # even_digit_keys + ['FIXATE']
	GW_keys 						= vision_keys + [compare_keys]


	vocabs['big vocab']				= create_vocab(D, [compare_keys]+[PRIM_keys]+digit_keys+['FIXATE'], rng)
	return vocabs