import os
import re
import sys
import json
import random
import numpy as np
import torch
import configparser
from tqdm import tqdm
from torch.autograd import Variable

USE_CUDA = True


class DatasetWoz3(object):
	'''
	data container for woz dataset
	'''
	def __init__(self, config, dataSplit_file, percentage=1.0, in_context=False):
		# setup
		feat_file = config['DATA']['feat_file']
		text_file = config['DATA']['text_file']
#		dataSplit_file = config['DATA']['dataSplit_file']
		vocab_file = config['DATA']['vocab_file']
		template_file = config['DATA']['template_file']
		if in_context:
			in_context_file = config['DATA']['in_context_file']
		else:
			in_context_file = None
		self.template = template_file # for further scoring
		self.in_context = in_context
		# hyper-params
		self.batch_size = config.getint('DATA', 'batch_size')
		self.percentage = percentage # percentage of data used
		self.data   = {'train':[],'valid':[],'test':[]} 
#		self.data   = {'train':[],'valid':[],'test_seen':[], 'test_unseen':[]} 
		self.data_index  = {'train': 0, 'valid': 0, 'test': 0} # index for accessing data
#		self.data_index  = {'train': 0, 'valid': 0, 'test_seen': 0, 'test_unseen': 0} # index for accessing data
		self.n_batch = {}
		self.shuffle = config.getboolean('DATA', 'shuffle')
		self.in_context = in_context
		# load vocab from file
		self._loadVocab(vocab_file) # a list of vocab, andy
		
		# set input feature cardinality
		self._setCardinality(template_file)
		self.do_size = self.dfs[1] - self.dfs[0]
		self.da_size = self.dfs[2] - self.dfs[1]
		self.sv_size = self.dfs[3] - self.dfs[2]
		
		# initialise dataset

		if self.in_context:
			self.in_context_data = {'train':[],'valid':[],'test':[]} 
			#self.in_context_data_index = {'train': 0, 'valid': 0, 'test': 0}
			self.in_context_n_batch = {}
		self._setupData(text_file, feat_file, dataSplit_file, in_context_file)

			#self._setUpInContextData()
		#self.reset()

 
	def reset(self):
		self.data_index  = {'train': 0, 'valid': 0, 'test': 0}
#		self.data_index  = {'train': 0, 'valid': 0, 'test_seen': 0, 'test_unseen': 0}
		if self.shuffle:
			random.shuffle(self.data['train'])

	def next_batch(self, data_type='train'):
		# reading a batch
		start = self.data_index[data_type]
		end = self.data_index[data_type] + self.batch_size
		data = self.data[data_type][start:end]
		self.data_index[data_type] += self.batch_size
		indexes = [i for i in range(start, end)]
		if self.in_context:
			# in_context_start = self.in_context_data_index[data_type]
			# in_context_end = self.in_context_data_index[data_type] + self.batch_size
			# in_context_data = self.in_context_data[data_type][in_context_start:in_context_end]
			# self.in_context_data_index[data_type] += self.batch_size
			in_context_data = self.in_context_data[data_type][start:end]

		if self.in_context:
			ori_data = self.get_data(data)
			#print(start, end, file=sys.stderr)
			#print(in_context_start, in_context_end, file=sys.stderr)
			#print(in_context_data, file=sys.stderr)
			i_data = self.get_data(in_context_data)
			return {'data': ori_data, 'in_context': i_data}
		else:
			return self.get_data(data)

#		return target_var, target_lengths, target_label, feats_var, words, featStrs, do_label, da_label, sv_label, sv_seqs

	def get_data(self, data):
		def indexes_from_sentence(sentence, add_eos=False):
			indexes = [self.word2index[word] if word in self.word2index else self.word2index['UNK_token'] for word in sentence.split(' ')]
			if add_eos:
				return indexes + [self.word2index['EOS_token']]
			else:
				return indexes

		# Pad a with the PAD symbol
		def pad_seq(seq, max_length):
			seq += [self.word2index['PAD_token'] for i in range(max_length - len(seq))]
			return seq

		# turn list of word indexes into 1-hot matrix
		def getOneHot(indexes):
			res = []
			for index in indexes:
				hot = [0]*len(self.word2index)
				hot[index] = 1
				res.append(hot)
			return res

		sentences, refs, feats, featStrs = [], [], [], []
#		do_label, da_label, sv_label, sv_seqs = [], [], [], []
		sv_indexes = []

		for dial_idx, turn_idx, text, meta in data:
			text_ori, text_delex = text['ori'], text['delex']
			sentences.append(indexes_from_sentence(text_delex, add_eos=True))
			refs.append(text_delex)

			# get semantic feature
			do_idx, da_idx, sv_idx, featStr = self.getFeatIdx(meta)
			do_cond = [1 if i in do_idx else 0 for i in range(self.do_size)] # domain condition
			da_cond = [1 if i in da_idx else 0 for i in range(self.da_size)] # dial act condition
			sv_cond = [1 if i in sv_idx else 0 for i in range(self.sv_size)] # slot/value condition
			feats.append(do_cond + da_cond + sv_cond)
			featStrs.append(featStr)

#			# get labels for da, slots
#			do_label.append(do_cond)
#			da_label.append(da_cond)
#			sv_label.append(sv_cond)
			sv_indexes.append(sv_idx)

		# Zip into pairs, sort by length (descending), unzip
		# Note: _words and _seqs should be sorted in the same order
		seq_pairs = sorted(zip(sentences, refs, feats, featStrs, sv_indexes), key=lambda p: len(p[0]), reverse=True)
		sentences, refs, feats, featStrs, sv_indexes = zip(*seq_pairs)

		# Pad with 0s to max length
		lengths = [len(s) for s in sentences]
		sentences_padded = [pad_seq(s, max(lengths)) for s in sentences]

		# Turn (batch_size, max_len) into (batch_size, max_len, n_vocab)
		sentences = [getOneHot(s) for s in sentences_padded]

		input_var = Variable(torch.FloatTensor(sentences))
		label_var = Variable(torch.LongTensor(sentences_padded))
		feats_var = Variable(torch.FloatTensor(feats))
#		do_label = Variable(torch.FloatTensor(do_label))
#		da_label = Variable(torch.FloatTensor(da_label))
#		sv_label = Variable(torch.FloatTensor(sv_label))

		if USE_CUDA:
			input_var = input_var.cuda()
			label_var = label_var.cuda()
			feats_var = feats_var.cuda()
#			do_label = do_label.cuda()
#			da_label = da_label.cuda()
#			sv_label = sv_label.cuda()

		return input_var, label_var, feats_var, lengths, refs, featStrs, sv_indexes

	def _setCardinality(self, template_file):
		self.cardinality = []
		with open(template_file) as f:
#			self.dfs = [0,0,0]
			self.dfs = [0,0,0,0]
			for line in f.readlines():
				self.cardinality.append(line.replace('\n',''))
				if line.startswith('d:'):
					self.dfs[1]+=1
				elif line.startswith('d-a:'):
					self.dfs[2]+=1
				elif line.startswith('d-a-s-v:'):
					self.dfs[3]+=1
			for i in range(0, len(self.dfs)-1):
				self.dfs[i+1] = self.dfs[i] + self.dfs[i+1]


	def printDataInfo(self):
		print('***** DATA INFO *****')
		print('Using {}% of training data'.format(self.percentage*100))
		print('BATCH SIZE:', self.batch_size)
		
		print('Train:', len(self.data['train']), 'turns')
		print('Valid:', len(self.data['valid']), 'turns')
		print('Test:', len(self.data['test']), 'turns')
#		print('Test (seen):', len(self.data['test_seen']))
#		print('Test (unseen):', len(self.data['test_unseen']))
		print('# of turns', file=sys.stderr)
		print('Train:', len(self.data['train']), file=sys.stderr)
		print('Valid:', len(self.data['valid']), file=sys.stderr)
		print('Test:', len(self.data['test']), file=sys.stderr)
#		print('Test (seen):', len(self.data['test_seen']), file=sys.stderr)
#		print('Test (unseen):', len(self.data['test_unseen']), file=sys.stderr)
		if self.in_context:
			print('Train:', len(self.in_context_data['train']), 'turns')
			print('Valid:', len(self.in_context_data['valid']), 'turns')
			print('Test:', len(self.in_context_data['test']), 'turns')
			print('# of in_context batches: Train {} Valid {} Test {}'.format(self.in_context_n_batch['train'], self.in_context_n_batch['valid'], self.in_context_n_batch['test']), file=sys.stderr)
			print('# of in_context batches: Train {} Valid {} Test {}'.format(self.in_context_n_batch['train'], self.in_context_n_batch['valid'], self.in_context_n_batch['test']))
		print('# of batches: Train {} Valid {} Test {}'.format(self.n_batch['train'], self.n_batch['valid'], self.n_batch['test']))
		print('# of batches: Train {} Valid {} Test {}'.format(self.n_batch['train'], self.n_batch['valid'], self.n_batch['test']), file=sys.stderr)
		print('*************************\n')


	def _setupData(self, text_file, feat_file, dataSplit_file, incontext_file):
		###modify this>???
		with open(text_file) as f:
			dial2text = json.load(f)
		with open(feat_file) as f:
			dial2meta = json.load(f)

		with open(dataSplit_file) as f:
			dataSet_split = json.load(f)
		
		with open(incontext_file) as f:
			das_collection = json.load(f)

		for data_type in ['train', 'valid', 'test']:
#		for data_type in ['train', 'valid', 'test_seen', 'test_unseen']:
			for dial_idx, turn_idx, _ in tqdm(dataSet_split[data_type], desc='Building data'):
				# might have empty feat turn which is not in feat file
				if turn_idx not in dial2meta[dial_idx]:
					continue

				meta = dial2meta[dial_idx][turn_idx]
				text = dial2text[dial_idx][turn_idx]
				self.data[data_type].append((dial_idx, turn_idx, text, meta))

				if self.in_context:
					in_context_dial, in_context_turn = get_similar(dial_idx, turn_idx, das_collection, dial2meta)

					in_context_meta = dial2meta[in_context_dial][in_context_turn]
					in_context_text = dial2text[in_context_dial][in_context_turn]
					self.in_context_data[data_type].append((in_context_dial, in_context_turn, in_context_text, in_context_meta))

		# percentage of training data
		if self.percentage < 1:
			_len = len(self.data['train'])
			self.data['train'] = self.data['train'][:int(_len*self.percentage)]
			if self.in_context:
				self.in_context_data['train'] = self.in_context_data['train'][:int(_len*self.percentage)]

		# setup number of batch
		for _type in ['train', 'valid', 'test']:
#		for _type in ['train', 'valid', 'test_seen', 'test_unseen']:
			self.n_batch[_type] = len(self.data[_type]) // self.batch_size
			if self.in_context:
				self.in_context_n_batch[_type] = len(self.in_context_data[_type]) // self.batch_size

		self.printDataInfo()
	
	def _loadVocab(self,vocab_file):
		# load vocab
		self.word2index = {}
		self.index2word = {}
		idx = 0
		with open(vocab_file) as fin:
			for word in fin.readlines():
				word = word.strip().split('\t')[0]
				self.word2index[word] = idx
				self.index2word[idx] = word
				idx += 1


	def getFeatIdx(self, meta):
		feat_container = []
		do_idx, da_idx, sv_idx = [], [], []
		for da, slots in meta.items():
			do = da.split('-')[0]
			_do_idx = self.cardinality.index('d:'+do) - self.dfs[0]
			if _do_idx not in do_idx:
				do_idx.append(_do_idx)
			da_idx.append( self.cardinality.index('d-a:'+da) - self.dfs[1] )
			for _slot in slots: # e.g. ('Day', '1', 'Wednesday ')
				sv_idx.append( self.cardinality.index('d-a-s-v:'+da+'-'+_slot[0]+'-'+_slot[1]) - self.dfs[2] )
				feat_container.append( da+'-'+_slot[0]+'-'+_slot[1] )

		feat_container = sorted(feat_container) # sort SVs across DAs to make sure universal order
		feat = '|'.join(feat_container)

		return do_idx, da_idx, sv_idx, feat

def get_similar(dial_idx, turn_idx, das_collection, feat):
	domains = ['Attraction', 'Hotel', 'Train', 'Taxi', 'Resaurant']
	meta = feat[dial_idx][turn_idx]
	other_than_general_booking = any([not(('general' in x) or ('Booking' in x)) for x in meta.keys()])
	other_than_general = any([not('general' in x) for x in meta.keys()])
	if other_than_general:
		if other_than_general_booking:
			for x in meta.keys():
				if (not('general' in x)) and not('Booking' in x):
					da = x
					s = set()
					for sv in meta[x]:
						s.add(sv[0])
					das = da + '-' + '-'.join(sorted(list(s)))
					break
		else:
			for x in meta.keys():
				if 'Booking' in x:
					da = x
					s = set()
					for sv in meta[x]:
						s.add(sv[0])
					das = da + '-' + '-'.join(sorted(list(s)))
					break
	else:
		for x in meta.keys():
			da = x
			s = set()
			for sv in meta[x]:
				s.add(sv[0])
			das = da + '-' + '-'.join(sorted(list(s)))
			break
	
	try:
		das = das.split('-')
	except:
		print(meta)
	while das != []:
		key = '-'.join(das)
		if key in das_collection.keys():
			if len(das_collection[key]) > 0:
				similar_dial, similar_turn = random.sample(das_collection[key], 1)[0]
				return similar_dial, similar_turn
			else:
				das = das[:-1]
		else:
			das = das[:-1] #back-off one slot

	return "PMUL1249","9" #none place holder    