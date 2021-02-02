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

def create_vocab(D, keys, rng=None):
	vocab = spa.Vocabulary(int(D), pointer_gen=rng, max_similarity=0)
	if type(keys) is list:
		vocab.populate(';'.join(keys))
	else:
		vocab.populate(keys)
	return vocab

def create_vocabs(D_GW, D_PRIM, seed=None):
	vocabs = {}
	rng = np.random.RandomState(seed)

	GW_keys 				= ['MORE ; LESS ; FIXATE'] + key_range(2,12,2) + ['D0=D2*~D2']
	PRIM_keys 				= ['GET_COM ; GET_ADD ; SET_COM ; SET_ADD ; SET_M'] + key_range(1,5,1)

	vocabs['GW']				= create_vocab(D_GW, GW_keys, rng)
	vocabs['PRIM']				= create_vocab(D_PRIM, PRIM_keys, rng)

	return vocabs