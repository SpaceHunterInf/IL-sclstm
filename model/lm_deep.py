import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from model.layers.decoder_deep import DecoderDeep
from model.layers.encoder_deep import EncoderDeep
from model.masked_cross_entropy import *
USE_CUDA = True

class LM_deep(nn.Module):
	def __init__(self, dec_type, input_size, output_size, hidden_size, d_size, n_layer=1, dropout=0.5, lr=0.001, sampling=False, in_context=False, k=6000):
		super(LM_deep, self).__init__()
		self.dec_type = dec_type
		self.hidden_size = hidden_size
		self.sampling = sampling
		self.b = 0
		self.k = k
		self.in_context = in_context
		print('Using deep version with {} layer'.format(n_layer))
		print('Using deep version with {} layer'.format(n_layer), file=sys.stderr)
		if self.in_context:
			self.enc = EncoderDeep(dec_type, input_size, output_size, hidden_size, d_size=d_size, n_layer=n_layer, dropout=dropout)
		self.dec = DecoderDeep(dec_type, input_size, output_size, hidden_size, d_size=d_size, n_layer=n_layer, dropout=dropout)
#		if self.dec_type != 'sclstm':
#			self.feat2hidden = nn.Linear(d_size, hidden_size)

		self.set_solver(lr)

	def	forward(self, input_var, dataset, feats_var, gen=False, beam_search=False, beam_size=1, context_var=None, context_feat=None):
		#print(input_var, input_var.shape)
		#print(feats_var, feats_var.shape)
		batch_size = dataset.batch_size
		self.b = (self.b + 1) #% batch_size #update_batch
		alpha = self.k / (self.k + torch.exp(torch.Tensor([self.b / self.k])))
		#print(alpha)
		if self.dec_type == 'sclstm':
			if self.in_context: #encode demonstration sample
				enc_init_indden = Variable(torch.zeros(batch_size, self.hidden_size))
				init_hidden = self.enc(context_var, dataset, init_hidden=enc_init_indden, init_feat=context_feat)
			else:
				init_hidden = Variable(torch.zeros(batch_size, self.hidden_size))
			if USE_CUDA:
				init_hidden = init_hidden.cuda()
			'''
			train/valid (gen=False, beam_search=False, beam_size=1)
	 		test w/o beam_search (gen=True, beam_search=False, beam_size=beam_size)
	 		test w/i beam_search (gen=True, beam_search=True, beam_size=beam_size)
			'''
			if beam_search:
				assert gen
				decoded_words = self.dec.beam_search(input_var, dataset, init_hidden=init_hidden, init_feat=feats_var, \
														gen=gen, beam_size=beam_size)
				return decoded_words # list (batch_size=1) of list (beam_size) with generated sentences

			# w/o beam_search
			sample_size = beam_size
			decoded_words = [ [] for _ in range(batch_size) ]
			for sample_idx in range(sample_size): # over generation
				self.output_prob, gens = self.dec(input_var, dataset, init_hidden=init_hidden, init_feat=feats_var, \
													gen=gen, sample_size=sample_size, sampling=self.sampling, alpha=alpha)
				#print(self.output_prob, gens)
				for batch_idx in range(batch_size):
					decoded_words[batch_idx].append(gens[batch_idx])
			#print(decoded_words)
			return decoded_words # list (batch_size) of list (sample_size) with generated sentences


		else: # TODO: vanilla lstm
			pass
#			last_hidden = self.feat2hidden(conds_batches)
#			self.output_prob, decoded_words = self.dec(input_seq, dataset, last_hidden=last_hidden, gen=gen, random_sample=self.random_sample)


	def set_solver(self, lr):
		if self.dec_type == 'sclstm':
			self.solver = torch.optim.Adam(self.dec.parameters(), lr=lr)
		else:
			self.solver = torch.optim.Adam([{'params': self.dec.parameters()}, {'params': self.feat2hidden.parameters()}], lr=lr)


	def get_loss(self, target_label, target_lengths):
		self.loss = masked_cross_entropy(
			self.output_prob.contiguous(), # -> batch x seq
			target_label.contiguous(), # -> batch x seq
			target_lengths)
		return self.loss
#		return {'rc': self.loss}


	def update(self, clip):
		# Back prop
		self.loss.backward()

		# Clip gradient norms
		_ = torch.nn.utils.clip_grad_norm(self.dec.parameters(), clip)

		# Update
		self.solver.step()

		# Zero grad
		self.solver.zero_grad()
