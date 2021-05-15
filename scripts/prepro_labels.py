"""
Preprocess a raw json dataset into hdf5/json files

Input: json file that has the form
[{ file_path: 'path/img.jpg', captions: ['a caption', ...] }, ...]

This script reads this json, does some basic preprocessing on the captions and encodes everything to arrays

Output: a json file and an hdf5 file
The hdf5 file contains several fields:
/labels is (M,max_length) uint32 array of encoded labels, zero padded
/label_start_ix and /label_end_ix are (N,) uint32 arrays of pointers to the 
  first and last indices (in range 1..M) of labels for each image
/label_length stores the length of the sequence for each of the M sequences

The json file has a dict that contains:
- an 'ix_to_word' field storing the vocab in form {ix:'word'}, where ix is 1-indexed
- an 'images' field that is a list holding auxiliary information for each image, 
  such as in particular the 'split' it was assigned to.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import json
import argparse
from random import shuffle, seed
import string
import h5py
import numpy as np
import torch
import torchvision.models as models
from PIL import Image
import re
from tqdm import tqdm

def build_vocab(imgs, imgs1, params):
    count_thr = params['word_count_threshold']

    # count up the number of words
    counts = {}

    for k, v in imgs.items():
        for s in v:
            sent = s
            sent_token = sent.split()
            for w in sent_token:
                counts[w] = counts.get(w, 0) + 1

    for k, v in imgs1.items():
        for s in v:
            sent = s
            sent_token = sent.split()
            for w in sent_token:
                counts[w] = counts.get(w, 0) + 1

    cw = sorted([(count, w) for w, count in counts.items()], reverse=True)
    print('top words and their counts:')
    print('\n'.join(map(str, cw)))

    # print some stats
    total_words = sum(counts.values())
    print('total words:', total_words)
    bad_words = [w for w, n in counts.items() if n <= count_thr]
    vocab = [w for w, n in counts.items() if n > count_thr]
    bad_count = sum(counts[w] for w in bad_words)
    print('number of bad words: %d/%d = %.2f%%' % (len(bad_words), len(counts), len(bad_words)*100.0/len(counts)))
    print('number of words in vocab would be %d' % (len(vocab), ))
    print('number of UNKs: %d/%d = %.2f%%' % (bad_count, total_words, bad_count*100.0/total_words))

    # lets look at the distribution of lengths as well
    sent_lengths = {}

    for k, v in imgs.items():
        for s in v:
            sent = s
            sent_token = sent.split()
            nw = len(sent_token)
            sent_lengths[nw] = sent_lengths.get(nw, 0) + 1

    for k, v in imgs1.items():
        for s in v:
            sent = s
            sent_token = sent.split()
            nw = len(sent_token)
            sent_lengths[nw] = sent_lengths.get(nw, 0) + 1

    max_len = max(sent_lengths.keys())
    print('max length sentence in raw data: ', max_len)
    print('sentence length distribution (count, number of words):')
    sum_len = sum(sent_lengths.values())
    for i in range(max_len+1):
        print('%2d: %10d   %f%%' % (i, sent_lengths.get(i,0), sent_lengths.get(i,0)*100.0/sum_len))

    # lets now produce the final annotations
    if bad_count > 0:
        # additional special UNK token we will use below to map infrequent words to
        print('inserting the special UNK token')
        vocab.append('UNK')

    return vocab

def encode_captions(imgs, imgs1, params, wtoi):

    max_length = params['max_length']
    N = len(imgs) + len(imgs1)
    M = N * 2

    label_arrays = []
    label_start_ix = np.zeros(N)  # note: these will be one-indexed
    label_end_ix = np.zeros(N)
    label_length = np.zeros(M)
    caption_counter = 0
    counter = 1

    n = 0
    for k, v in imgs.items():
        Li = np.zeros((2, max_length))
        vn = len(v)
        s1 = v[0]
        if vn >= 2:
            for j, s in enumerate(v[:2]):
                sent = s
                sent_token = sent.split()
                label_length[caption_counter] = min(max_length, len(sent_token))
                caption_counter += 1
                for i, w in enumerate(sent_token):
                    if i < max_length:
                        Li[j, i] = wtoi[w]
        else:
            for j, s in enumerate(v):
                sent = s
                sent_token = sent.split()
                label_length[caption_counter] = min(max_length, len(sent_token))
                caption_counter += 1
                for i, w in enumerate(sent_token):
                    if i < max_length:
                        Li[j, i] = wtoi[w]
            sent_token = s1.split()
            for i in range(2 - vn):
                label_length[caption_counter] = min(max_length, len(sent_token))
                caption_counter += 1
            for i, w in enumerate(sent_token):
                if i < max_length:
                    for x in range(vn, 2):
                        Li[x, i] = wtoi[w]
        # note: word indices are 1-indexed, and captions are padded with zeros
        label_arrays.append(Li)
        label_start_ix[n] = counter
        label_end_ix[n] = counter + 2 - 1
        counter += 2
        n += 1

    for k, v in imgs1.items():
        Li = np.zeros((2, max_length))
        vn = len(v)
        s1 = v[0]
        if vn >= 2:
            for j, s in enumerate(v[:2]):
                sent = s
                sent_token = sent.split()
                label_length[caption_counter] = min(max_length, len(sent_token))
                caption_counter += 1
                for i, w in enumerate(sent_token):
                    if i < max_length:
                        Li[j, i] = wtoi[w]
        else:
            for j, s in enumerate(v):
                sent = s
                sent_token = sent.split()
                label_length[caption_counter] = min(max_length, len(sent_token))
                caption_counter += 1
                for i, w in enumerate(sent_token):
                    if i < max_length:
                        Li[j, i] = wtoi[w]
            sent_token = s1.split()
            for i in range(2 - vn):
                label_length[caption_counter] = min(max_length, len(sent_token))
                caption_counter += 1
            for i, w in enumerate(sent_token):
                if i < max_length:
                    for x in range(vn, 2):
                        Li[x, i] = wtoi[w]
        # note: word indices are 1-indexed, and captions are padded with zeros
        label_arrays.append(Li)
        label_start_ix[n] = counter
        label_end_ix[n] = counter + 2 - 1
        counter += 2
        n += 1

    L = np.concatenate(label_arrays, axis=0) # put all the labels together
    # assert L.shape[0] == M, 'lengths don\'t match? that\'s weird'
    assert L.shape[0] == M, 'lengths don\'t match? that\'s weird'
    assert np.all(label_length > 0), 'error: some caption had no words?'

    print('encoded captions to array of size ', L.shape)
    return L, label_start_ix, label_end_ix, label_length

def main(params):

    imgs = json.load(open(params['input_json'], 'r'))
    imgs1 = json.load(open(params['input1_json'], 'r'))

    seed(123) # make reproducible
    
    # create the vocab
    vocab = build_vocab(imgs, imgs1, params)
    itow = {i+1: w for i, w in enumerate(vocab)} # a 1-indexed vocab translation table
    wtoi = {w: i+1 for i, w in enumerate(vocab)} # inverse table
    
    # encode captions in large arrays, ready to ship to hdf5 file
    L, label_start_ix, label_end_ix, label_length = encode_captions(imgs, imgs1, params, wtoi)

    # create output h5 file
    N = len(imgs) + len(imgs1)
    f_lb = h5py.File(params['output_h5']+'_label.h5', "w")
    f_lb.create_dataset("labels", dtype='uint32', data=L)
    f_lb.create_dataset("label_start_ix", dtype='uint32', data=label_start_ix)
    f_lb.create_dataset("label_end_ix", dtype='uint32', data=label_end_ix)
    f_lb.create_dataset("label_length", dtype='uint32', data=label_length)
    f_lb.close()

    # create output json file
    out = {}
    out['ix_to_word'] = itow # encode the (1-indexed) vocab
    out['images'] = []

    splits = json.load(open(params['splits'], 'r'))
    train = splits['train']
    val = splits['val']
    test = splits['test']
    for k, v in tqdm(imgs.items()):
        jimg = {}
        k_1 = k.replace('default', 'semantic')
        jimg['file_path_default'] = k
        jimg['file_path_semantic'] = k_1
        jimg['sentence'] = v
        jimg['id'] = int(re.sub("\D", "", k))
        jimg['width'] = 480
        jimg['height'] = 320

        if int(re.sub("\D", "", k)) in train:
            jimg['split'] = "train"
        elif int(re.sub("\D", "", k)) in val:
            jimg['split'] = "val"
        else:
            jimg['split'] = "test"
        out['images'].append(jimg)

    for k, v in tqdm(imgs1.items()):
        jimg = {}
        k_1 = k.replace('default', 'nonsemantic')
        jimg['file_path_default'] = k
        jimg['file_path_nonsemantic'] = k_1
        jimg['sentence'] = v
        jimg['id'] = int(re.sub("\D", "", k)) + 40000
        jimg['width'] = 480
        jimg['height'] = 320

        if int(re.sub("\D", "", k)) in train:
            jimg['split'] = "train"
        elif int(re.sub("\D", "", k)) in val:
            jimg['split'] = "val"
        else:
            jimg['split'] = "test"
        out['images'].append(jimg)

    json.dump(out, open(params['output_json'], 'w'))
    print('wrote ', params['output_json'])

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # input json
    parser.add_argument('--input_json', type=str, default='', help='input json file of change to process into hdf5')
    parser.add_argument('--input1_json', type=str, default='', help='input json file of nochange to process into hdf5')
    parser.add_argument('--output_json', type=str, default='', help='output json file')
    parser.add_argument('--output_h5', type=str, default='', help='output h5 file')
    parser.add_argument('--images_root', type=str, default='', help='root location in which images are stored, to be prepended to file_path in input json')
    parser.add_argument('--splits', type=str, default='', help='splits.json file location')
    # options
    parser.add_argument('--max_length', default=20, type=int, help='max length of a caption, in number of words. captions longer than this get clipped.')
    parser.add_argument('--word_count_threshold', default=0, type=int, help='only words that occur more than this number of times will be put in vocab')

    args = parser.parse_args()
    params = vars(args) # convert to ordinary dict
    print('parsed input parameters:')
    print(json.dumps(params, indent=2))
    main(params)
