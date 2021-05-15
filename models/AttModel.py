from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import misc.utils as utils

from .CaptionModel import CaptionModel
from .BANModel import Ban
import time

bad_endings = []

class AttModel(CaptionModel):
    def __init__(self, opt):
        super(AttModel, self).__init__()
        self.vocab_size = opt.vocab_size
        self.input_encoding_size = opt.input_encoding_size
        self.rnn_size = opt.rnn_size
        self.num_layers = opt.num_layers
        self.drop_prob_lm = opt.drop_prob_lm
        self.seq_length = getattr(opt, 'max_length', 22) or opt.seq_length
        self.att_feat_size = opt.att_feat_size
        self.att_hid_size = opt.att_hid_size
        self.use_bn = getattr(opt, 'use_bn', 0)
        self.ss_prob = 0.0 # Schedule sampling probability
        self.embed = nn.Embedding(self.vocab_size + 1, self.input_encoding_size)
        embed = torch.load("./data/embedding/embedding.pt")
        self.embed.weight = nn.Parameter(embed)
        self.embed.weight.requires_grad = True
        self.att_embed = nn.Sequential(*(
                                    ((nn.BatchNorm1d(self.att_feat_size),) if self.use_bn else ())+
                                    (nn.Linear(self.att_feat_size, 1024),
                                    nn.ReLU(),
                                    nn.Dropout(self.drop_prob_lm))+
                                    ((nn.BatchNorm1d(1024),) if self.use_bn == 2 else ())))

        self.logit_layers = getattr(opt, 'logit_layers', 1)
        if self.logit_layers == 1:
            self.logit = nn.Linear(self.rnn_size, self.vocab_size + 1)
        else:
            self.logit = [[nn.Linear(self.rnn_size, self.rnn_size), nn.ReLU(), nn.Dropout(0.5)] for _ in range(opt.logit_layers - 1)]
            self.logit = nn.Sequential(*(reduce(lambda x, y: x+y, self.logit) + [nn.Linear(self.rnn_size, self.vocab_size + 1)]))
        self.ctx2att = nn.Linear(self.rnn_size, self.att_hid_size)

        # For remove bad endding
        self.vocab = opt.vocab
        self.bad_endings_ix = [int(k) for k, v in self.vocab.items() if v in bad_endings]

        self.a = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.a.data.fill_(0.5)

    def init_hidden(self, bsz):
        weight = next(self.parameters())
        return (weight.new_zeros(self.num_layers, bsz, self.rnn_size),
                weight.new_zeros(self.num_layers, bsz, self.rnn_size))

    def box_feat(self, boxes, dim):

        batch_size = boxes.shape[0]
        num_boxes = boxes.shape[1]

        # position
        pos = boxes.new_zeros((batch_size, num_boxes, 4))
        pos[:, :, 0] = boxes[:, :, 0] * 100
        pos[:, :, 1] = boxes[:, :, 1] * 100
        pos[:, :, 2] = boxes[:, :, 2] * 100
        pos[:, :, 3] = boxes[:, :, 3] * 100

        # sin/cos embedding
        dim_mat = 1000 ** (torch.arange(dim, dtype=boxes.dtype, device=boxes.device) / dim)
        sin_embedding = (pos.view((batch_size, num_boxes, 4, 1)) / dim_mat.view((1, 1, 1, -1))).sin()
        cos_embedding = (pos.view((batch_size, num_boxes, 4, 1)) / dim_mat.view((1, 1, 1, -1))).cos()

        return torch.cat((sin_embedding, cos_embedding), dim=-1)

    def att_feat(self, semantic_feat, semantic1_feat, att_feats, att1_feats, box_feat, box1_feat):

        batch_size = att_feats.size(0)

        s_f = att_feats
        s1_f = att1_feats

        nums = semantic_feat.shape[1]
        semantic_feat = semantic_feat.view(batch_size * nums, semantic_feat.shape[2])
        semantic_feat = self.embed(semantic_feat.long())
        semantic_feat = semantic_feat.view(semantic_feat.shape[0], semantic_feat.shape[1] * semantic_feat.shape[2])
        semantic_feat = semantic_feat.view(batch_size, nums, semantic_feat.shape[1])

        semantic1_feat = semantic1_feat.view(batch_size * nums, semantic1_feat.shape[2])
        semantic1_feat = self.embed(semantic1_feat.long())
        semantic1_feat = semantic1_feat.view(semantic1_feat.shape[0], semantic1_feat.shape[1] * semantic1_feat.shape[2])
        semantic1_feat = semantic1_feat.view(batch_size, nums, semantic1_feat.shape[1])

        ban = Ban().cuda()
        first = 0
        last = 300
        for i in range(4):
            s = semantic_feat[:, :, first:last]
            s1 = semantic1_feat[:, :, first:last]
            first += 300
            last += 300

            s_f = ban(s_f, s)
            s1_f = ban(s1_f, s1)

        b_f = self.box_feat(box_feat, 256)
        b1_f = self.box_feat(box1_feat, 256)
        b_f = b_f.view(batch_size, nums, 4 * 512)
        b1_f = b1_f.view(batch_size, nums, 4 * 512)

        semantic_feat = s_f + b_f
        semantic1_feat = s1_f + b1_f

        # cosin computation -- semantic_feat
        sf = semantic_feat.unsqueeze(2)
        sf = sf.repeat(1, 1, nums, 1)
        sf = sf.view(batch_size, nums * nums, sf.shape[3])
        sf1 = semantic1_feat.unsqueeze(1)
        sf1 = sf1.repeat(1, nums, 1, 1)
        sf1 = sf1.view(batch_size, nums * nums, sf1.shape[3])
        cosResult = torch.cosine_similarity(sf, sf1, dim=2)
        cosResult = cosResult.view(batch_size, nums, nums)
        cosResult = torch.max(cosResult, dim=2)
        cosResult = cosResult[0]

        sf_ = semantic1_feat.unsqueeze(2)
        sf_ = sf_.repeat(1, 1, nums, 1)
        sf_ = sf_.view(batch_size, nums * nums, sf_.shape[3])
        sf1_ = semantic_feat.unsqueeze(1)
        sf1_ = sf1_.repeat(1, nums, 1, 1)
        sf1_ = sf1_.view(batch_size, nums * nums, sf1_.shape[3])
        cosResult1 = torch.cosine_similarity(sf_, sf1_, dim=2)
        cosResult1 = cosResult1.view(batch_size, nums, nums)
        cosResult1 = torch.max(cosResult1, dim=2)
        cosResult1 = cosResult1[0]

        one = torch.ones((batch_size, nums, 1)).cuda()
        cosResult = cosResult.unsqueeze(2)
        weight = self.a * (one - cosResult)


        cosResult1 = cosResult1.unsqueeze(2)
        weight1 = self.a * (one - cosResult1)

        new_semantic_feat = torch.mul(semantic_feat, weight)
        new_semantic1_feat = torch.mul(semantic1_feat, weight1)

        return new_semantic_feat, new_semantic1_feat

    def _forward(self, semantic_feat, semantic1_feat, att_feats, att1_feats, box_feat, box1_feat, seq):
        batch_size = att_feats.size(0)
        if seq.ndim == 3:
            seq = seq.reshape(-1, seq.shape[2])
        seq_per_img = seq.shape[0] // batch_size
        state = self.init_hidden(batch_size*seq_per_img)
        outputs = att_feats.new_zeros(batch_size*seq_per_img, seq.size(1) - 1, self.vocab_size+1)

        # att_feat
        new_semantic_feat, new_semantic1_feat = self.att_feat(semantic_feat, semantic1_feat, att_feats, att1_feats, box_feat, box1_feat)

        if seq_per_img > 1:
            new_semantic_feat, new_semantic1_feat = utils.repeat_tensors(seq_per_img,
                                                                                      [new_semantic_feat, new_semantic1_feat])


        for i in range(seq.size(1) - 1):
            if self.training and i >= 1 and self.ss_prob > 0.0: # otherwiste no need to sample
                sample_prob = att_feats.new(batch_size*seq_per_img).uniform_(0, 1)
                sample_mask = sample_prob < self.ss_prob
                if sample_mask.sum() == 0:
                    it = seq[:, i].clone()
                else:
                    sample_ind = sample_mask.nonzero().view(-1)
                    it = seq[:, i].data.clone()
                    prob_prev = torch.exp(outputs[:, i-1].detach()) # fetch prev distribution: shape Nx(M+1)
                    it.index_copy_(0, sample_ind, torch.multinomial(prob_prev, 1).view(-1).index_select(0, sample_ind))
            else:
                it = seq[:, i].clone()
            # break if all the sequences end
            if i >= 1 and seq[:, i].sum() == 0:
                break

            output, state = self.get_logprobs_state(it, new_semantic_feat, new_semantic1_feat, state)
            outputs[:, i] = output

        return outputs

    def get_logprobs_state(self, it, new_semantic_feat, new_semantic1_feat, state, output_logsoftmax=1):
        # 'it' contains a word index
        xt = self.embed(it)

        output, state = self.core(xt, new_semantic_feat, new_semantic1_feat, state)
        if output_logsoftmax:
            logprobs = F.log_softmax(self.logit(output), dim=1)
        else:
            logprobs = self.logit(output)

        return logprobs, state

    def _old_sample_beam(self, semmantic_feat, semantic1_feat, att_feats, att1_feat, box_feat, box1_feat, opt={}):
        beam_size = opt.get('beam_size', 10)
        group_size = opt.get('group_size', 1)
        sample_n = opt.get('sample_n', 10)
        # when sample_n == beam_size then each beam is a sample.
        assert sample_n == 1 or sample_n == beam_size // group_size, 'when beam search, sample_n == 1 or beam search'
        batch_size = att_feats.size(0)

        new_semantic_feat, new_semantic1_feat = self.att_feat(semmantic_feat, semantic1_feat, att_feats, att1_feat, box_feat, box1_feat)

        assert beam_size <= self.vocab_size + 1, 'lets assume this for now, otherwise this corner case causes a few headaches down the road. can be dealt with in future if needed'
        seq = att_feats.new_zeros((batch_size*sample_n, self.seq_length), dtype=torch.long)
        seqLogprobs = att_feats.new_zeros(batch_size*sample_n, self.seq_length, self.vocab_size + 1)
        # lets process every image independently for now, for simplicity

        self.done_beams = [[] for _ in range(batch_size)]
        for k in range(batch_size):
            state = self.init_hidden(beam_size)
            tmp_new_semantic_feat, tmp_new_semantic1_feat = utils.repeat_tensors(beam_size,
                                                                                [new_semantic_feat[k:k + 1],
                                                                                new_semantic1_feat[k:k + 1]
                                                                                ]
                                                                                )

            for t in range(1):
                if t == 0: # input <bos>
                    it = att_feats.new_zeros([beam_size], dtype=torch.long)

                logprobs, state = self.get_logprobs_state(it, tmp_new_semantic_feat, tmp_new_semantic1_feat, state)

            self.done_beams[k] = self.old_beam_search(state, logprobs, tmp_new_semantic_feat, tmp_new_semantic1_feat, opt=opt)
            if sample_n == beam_size:
                for _n in range(sample_n):
                    seq[k*sample_n+_n, :] = self.done_beams[k][_n]['seq']
                    seqLogprobs[k*sample_n+_n, :] = self.done_beams[k][_n]['logps']
            else:
                seq[k, :] = self.done_beams[k][0]['seq'] # the first beam has highest cumulative score
                seqLogprobs[k, :] = self.done_beams[k][0]['logps']
        # return the samples and their log likelihoods
        return seq, seqLogprobs

    def _sample_beam(self, semmantic_feat, semantic1_feat, att_feats, att1_feat, box_feat, box1_feat, opt={}):
        beam_size = opt.get('beam_size', 10)
        group_size = opt.get('group_size', 1)
        sample_n = opt.get('sample_n', 10)
        # when sample_n == beam_size then each beam is a sample.
        assert sample_n == 1 or sample_n == beam_size // group_size, 'when beam search, sample_n == 1 or beam search'
        batch_size = att_feats.size(0)

        new_semantic_feat, new_semantic1_feat = self.att_feat(semmantic_feat, semantic1_feat, att_feats, att1_feat, box_feat, box1_feat)

        assert beam_size <= self.vocab_size + 1, 'lets assume this for now, otherwise this corner case causes a few headaches down the road. can be dealt with in future if needed'
        seq = att_feats.new_zeros((batch_size*sample_n, self.seq_length), dtype=torch.long)
        seqLogprobs = att_feats.new_zeros(batch_size*sample_n, self.seq_length, self.vocab_size + 1)
        # lets process every image independently for now, for simplicity

        self.done_beams = [[] for _ in range(batch_size)]
        
        state = self.init_hidden(batch_size)

        # first step, feed bos
        it = att_feats.new_zeros([batch_size], dtype=torch.long)
        logprobs, state = self.get_logprobs_state(it, new_semantic_feat, new_semantic1_feat, state)

        new_semantic_feat, new_semantic1_feat = utils.repeat_tensors(
            beam_size,
            [new_semantic_feat, new_semantic1_feat]
            )
        self.done_beams = self.beam_search(state, logprobs, new_semantic_feat, new_semantic1_feat, opt=opt)
        for k in range(batch_size):
            if sample_n == beam_size:
                for _n in range(sample_n):
                    seq_len = self.done_beams[k][_n]['seq'].shape[0]
                    seq[k*sample_n+_n, :seq_len] = self.done_beams[k][_n]['seq']
                    seqLogprobs[k*sample_n+_n, :seq_len] = self.done_beams[k][_n]['logps']
            else:
                seq_len = self.done_beams[k][0]['seq'].shape[0]
                seq[k, :seq_len] = self.done_beams[k][0]['seq'] # the first beam has highest cumulative score
                seqLogprobs[k, :seq_len] = self.done_beams[k][0]['logps']
        # return the samples and their log likelihoods
        return seq, seqLogprobs

    def _sample(self, semmantic_feat, semantic1_feat, att_feats, att1_feat, box_feat, box1_feat, opt={}):

        sample_method = opt.get('sample_method', 'greedy')
        beam_size = opt.get('beam_size', 1)
        temperature = opt.get('temperature', 1.0)
        sample_n = int(opt.get('sample_n', 1))
        group_size = opt.get('group_size', 1)
        output_logsoftmax = opt.get('output_logsoftmax', 1)
        decoding_constraint = opt.get('decoding_constraint', 0)
        block_trigrams = opt.get('block_trigrams', 0)
        remove_bad_endings = opt.get('remove_bad_endings', 0)
        if beam_size > 1 and sample_method in ['greedy', 'beam_search']:
            return self._sample_beam(semmantic_feat, semantic1_feat, att_feats, att1_feat, box_feat, box1_feat, opt)
        if group_size > 1:
            return self._diverse_sample(semmantic_feat, semantic1_feat, att_feats, att1_feat, box_feat, box1_feat, opt)

        batch_size = att_feats.size(0)
        state = self.init_hidden(batch_size*sample_n)

        new_semantic_feat, new_semantic1_feat = self.att_feat(semmantic_feat, semantic1_feat, att_feats, att1_feat, box_feat, box1_feat)

        if sample_n > 1:
            new_semantic_feat, new_semantic1_feat = utils.repeat_tensors(
                sample_n,
                [new_semantic_feat,
                 new_semantic1_feat
                 ]
            )

        trigrams = [] # will be a list of batch_size dictionaries
        
        seq = att_feats.new_zeros((batch_size*sample_n, self.seq_length), dtype=torch.long)
        seqLogprobs = att_feats.new_zeros(batch_size*sample_n, self.seq_length, self.vocab_size + 1)
        for t in range(self.seq_length + 1):
            if t == 0: # input <bos>
                it = att_feats.new_zeros(batch_size*sample_n, dtype=torch.long)
            logprobs, state = self.get_logprobs_state(it, new_semantic_feat, new_semantic1_feat, state, output_logsoftmax=output_logsoftmax)

            if decoding_constraint and t > 0:
                tmp = logprobs.new_zeros(logprobs.size())
                tmp.scatter_(1, seq[:,t-1].data.unsqueeze(1), float('-inf'))
                logprobs = logprobs + tmp

            if remove_bad_endings and t > 0:
                tmp = logprobs.new_zeros(logprobs.size())
                prev_bad = np.isin(seq[:,t-1].data.cpu().numpy(), self.bad_endings_ix)
                # Make it impossible to generate bad_endings
                tmp[torch.from_numpy(prev_bad.astype('uint8')), 0] = float('-inf')
                logprobs = logprobs + tmp

            if block_trigrams and t >= 3:
                # Store trigram generated at last step
                prev_two_batch = seq[:,t-3:t-1]
                for i in range(batch_size):
                    prev_two = (prev_two_batch[i][0].item(), prev_two_batch[i][1].item())
                    current  = seq[i][t-1]
                    if t == 3: # initialize
                        trigrams.append({prev_two: [current]}) # {LongTensor: list containing 1 int}
                    elif t > 3:
                        if prev_two in trigrams[i]: # add to list
                            trigrams[i][prev_two].append(current)
                        else: # create list
                            trigrams[i][prev_two] = [current]
                # Block used trigrams at next step
                prev_two_batch = seq[:,t-2:t]
                mask = torch.zeros(logprobs.size(), requires_grad=False).cuda() # batch_size x vocab_size
                for i in range(batch_size):
                    prev_two = (prev_two_batch[i][0].item(), prev_two_batch[i][1].item())
                    if prev_two in trigrams[i]:
                        for j in trigrams[i][prev_two]:
                            mask[i,j] += 1
                # Apply mask to log probs
                #logprobs = logprobs - (mask * 1e9)
                alpha = 2.0 # = 4
                logprobs = logprobs + (mask * -0.693 * alpha) # ln(1/2) * alpha (alpha -> infty works best)

            # sample the next word
            if t == self.seq_length: # skip if we achieve maximum length
                break
            it, sampleLogprobs = self.sample_next_word(logprobs, sample_method, temperature)

            # stop when all finished
            if t == 0:
                unfinished = it > 0
            else:
                unfinished = unfinished * (it > 0)
            it = it * unfinished.type_as(it)
            seq[:,t] = it
            seqLogprobs[:,t] = logprobs
            # quit loop if all sequences have finished
            if unfinished.sum() == 0:
                break

        return seq, seqLogprobs

    def _diverse_sample(self, semmantic_feat, semantic1_feat, att_feats, att1_feat, box_feat, box1_feat, opt={}):

        sample_method = opt.get('sample_method', 'greedy')
        beam_size = opt.get('beam_size', 1)
        temperature = opt.get('temperature', 1.0)
        group_size = opt.get('group_size', 1)
        diversity_lambda = opt.get('diversity_lambda', 0.5)
        decoding_constraint = opt.get('decoding_constraint', 0)
        block_trigrams = opt.get('block_trigrams', 0)
        remove_bad_endings = opt.get('remove_bad_endings', 0)

        batch_size = att_feats.size(0)
        state = self.init_hidden(batch_size)

        new_semantic_feat, new_semantic1_feat = self.att_feat(semmantic_feat, semantic1_feat, att_feats, att1_feat, box_feat, box1_feat)

        trigrams_table = [[] for _ in range(group_size)] # will be a list of batch_size dictionaries

        seq_table = [att_feats.new_zeros((batch_size, self.seq_length), dtype=torch.long) for _ in range(group_size)]
        seqLogprobs_table = [att_feats.new_zeros(batch_size, self.seq_length) for _ in range(group_size)]
        state_table = [self.init_hidden(batch_size) for _ in range(group_size)]

        for tt in range(self.seq_length + group_size):
            for divm in range(group_size):
                t = tt - divm
                seq = seq_table[divm]
                seqLogprobs = seqLogprobs_table[divm]
                trigrams = trigrams_table[divm]
                if t >= 0 and t <= self.seq_length-1:
                    if t == 0: # input <bos>
                        it = att_feats.new_zeros(batch_size, dtype=torch.long)
                    else:
                        it = seq[:, t-1] # changed

                    logprobs, state_table[divm] = self.get_logprobs_state(it, new_semantic_feat, new_semantic1_feat,
                                                                          state_table[divm])
                    logprobs = F.log_softmax(logprobs / temperature, dim=-1)

                    # Add diversity
                    if divm > 0:
                        unaug_logprobs = logprobs.clone()
                        for prev_choice in range(divm):
                            prev_decisions = seq_table[prev_choice][:, t]
                            logprobs[:, prev_decisions] = logprobs[:, prev_decisions] - diversity_lambda
                    
                    if decoding_constraint and t > 0:
                        tmp = logprobs.new_zeros(logprobs.size())
                        tmp.scatter_(1, seq[:,t-1].data.unsqueeze(1), float('-inf'))
                        logprobs = logprobs + tmp

                    if remove_bad_endings and t > 0:
                        tmp = logprobs.new_zeros(logprobs.size())
                        prev_bad = np.isin(seq[:,t-1].data.cpu().numpy(), self.bad_endings_ix)
                        # Impossible to generate remove_bad_endings
                        tmp[torch.from_numpy(prev_bad.astype('uint8')), 0] = float('-inf')
                        logprobs = logprobs + tmp

                    # Mess with trigrams
                    if block_trigrams and t >= 3:
                        # Store trigram generated at last step
                        prev_two_batch = seq[:,t-3:t-1]
                        for i in range(batch_size): # = seq.size(0)
                            prev_two = (prev_two_batch[i][0].item(), prev_two_batch[i][1].item())
                            current  = seq[i][t-1]
                            if t == 3: # initialize
                                trigrams.append({prev_two: [current]}) # {LongTensor: list containing 1 int}
                            elif t > 3:
                                if prev_two in trigrams[i]: # add to list
                                    trigrams[i][prev_two].append(current)
                                else: # create list
                                    trigrams[i][prev_two] = [current]
                        # Block used trigrams at next step
                        prev_two_batch = seq[:,t-2:t]
                        mask = torch.zeros(logprobs.size(), requires_grad=False).cuda() # batch_size x vocab_size
                        for i in range(batch_size):
                            prev_two = (prev_two_batch[i][0].item(), prev_two_batch[i][1].item())
                            if prev_two in trigrams[i]:
                                for j in trigrams[i][prev_two]:
                                    mask[i,j] += 1
                        # Apply mask to log probs
                        #logprobs = logprobs - (mask * 1e9)
                        alpha = 2.0 # = 4
                        logprobs = logprobs + (mask * -0.693 * alpha) # ln(1/2) * alpha (alpha -> infty works best)

                    it, sampleLogprobs = self.sample_next_word(logprobs, sample_method, 1)

                    # stop when all finished
                    if t == 0:
                        unfinished = it > 0
                    else:
                        unfinished = (seq[:,t-1] > 0) & (it > 0) # changed
                    it = it * unfinished.type_as(it)
                    seq[:,t] = it
                    seqLogprobs[:,t] = sampleLogprobs.view(-1)

        return torch.stack(seq_table, 1).reshape(batch_size * group_size, -1), torch.stack(seqLogprobs_table, 1).reshape(batch_size * group_size, -1)

class UpDownCore(nn.Module):
    def __init__(self, opt, use_maxout=False):
        super(UpDownCore, self).__init__()
        self.drop_prob_lm = opt.drop_prob_lm
        self.rnn_size = opt.rnn_size
        self.input_encoding_size = opt.input_encoding_size
        self.w = nn.Sequential(nn.Linear(36, 1),
                               nn.ReLU(),
                               nn.Dropout(self.drop_prob_lm))

        self.att_lstm = nn.LSTMCell(self.input_encoding_size + opt.rnn_size + 2048*2, opt.rnn_size)
        self.lang_lstm = nn.LSTMCell(2048*2 + self.rnn_size, opt.rnn_size)
        self.attention = Attention(opt)

    def forward(self, xt, new_semantic_feat, new_semantic1_feat, state):
        prev_h = state[0][-1]
        batch_size = new_semantic_feat.shape[0]
        nums = new_semantic_feat.shape[1]
        d_feat = new_semantic_feat.shape[2]

        fc_feat = new_semantic_feat.mean(1)
        fc1_feat = new_semantic1_feat.mean(1)

        new_fc_feat = torch.cat([fc_feat, fc1_feat], 1)

        att_lstm_input = torch.cat([prev_h, new_fc_feat, xt], 1)

        h_att, c_att = self.att_lstm(att_lstm_input, (state[0][0], state[1][0]))

        att = self.attention(h_att, new_semantic_feat, new_semantic1_feat)

        lang_lstm_input = torch.cat([att, h_att], 1)

        h_lang, c_lang = self.lang_lstm(lang_lstm_input, (state[0][1], state[1][1]))

        output = F.dropout(h_lang, self.drop_prob_lm, self.training)
        state = (torch.stack([h_att, h_lang]), torch.stack([c_att, c_lang]))

        return output, state

class Attention(nn.Module):
    def __init__(self, opt):
        super(Attention, self).__init__()
        self.rnn_size = opt.rnn_size
        self.att_hid_size = opt.att_hid_size

        self.h2att = nn.Linear(self.rnn_size, self.att_hid_size)
        self.alpha_net = nn.Linear(self.att_hid_size, 1)
        self.W = nn.Linear(self.att_hid_size * 3, self.att_hid_size)
        self.WV = nn.Linear(2048*2, self.att_hid_size)

    def forward(self, h, new_semantic_feat, new_semantic1_feat):

        batch_size = new_semantic_feat.shape[0]
        nums = new_semantic_feat.shape[1]

        feat = torch.cat([new_semantic_feat, new_semantic1_feat], dim=2)
        att_feat = self.WV(feat)

        att_h = self.h2att(h)
        att_h = att_h.unsqueeze(1).expand_as(att_feat)
        dot = att_feat + att_h
        dot = torch.tanh(dot)
        dot = dot.view(-1, self.att_hid_size)
        dot = self.alpha_net(dot)
        dot = dot.view(-1, nums)
        weight = F.softmax(dot, dim=1)
        att_res = torch.bmm(weight.unsqueeze(1), feat).squeeze(1)

        return att_res

class UpDownModel(AttModel):
    def __init__(self, opt):
        super(UpDownModel, self).__init__(opt)
        self.num_layers = 2
        self.core = UpDownCore(opt)







