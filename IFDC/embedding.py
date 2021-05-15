import json
import argparse
import six
import numpy as np
import torch
from tqdm import tqdm

def get_vocabs(dict_file):
    with open(dict_file, "r", encoding="utf-8") as f:
        unit = json.load(f)
        vocab = unit["ix_to_word"]
        print(vocab)
    return vocab

def read_embeddings(emb_file):
    embs = dict()
    with open(emb_file, "rb") as f:
        for i, line in tqdm(enumerate(f)):
            if not line:
                break
            if len(line) == 0:
                continue
            l_split = line.decode('utf8').strip().split(' ')
            if len(l_split) == 2:
                continue
            embs[l_split[0]] = [float(em) for em in l_split[1:]]
    return embs

def match_embeddings(vocab, emb):
    dim = len(six.next(six.itervalues(emb)))
    filtered_embeddings = np.zeros((len(vocab)+1, dim))
    for w_id, w in vocab.items():
        w_id = int(w_id)
        if w in emb:
            filtered_embeddings[w_id] = emb[w]
    return torch.Tensor(filtered_embeddings)

def main():

    parser = argparse.ArgumentParser(description="embeddings_to_torch.py")
    parser.add_argument("-emb_file", type=str, default="./data/embedding/glove.6B.300d.txt")
    parser.add_argument("-output_file", type=str, default="./data/embedding/embedding_spot_1")
    parser.add_argument("-dict_file", type=str, default="./data/output_train.json")
    parser.add_argument("-verbose", action="store_true", default=False)
    opt = parser.parse_args()

    vocab = get_vocabs(opt.dict_file)

    vectors = read_embeddings(opt.emb_file)

    filtered_embeddings = match_embeddings(vocab, vectors)

    of = opt.output_file + ".pt"
    torch.save(filtered_embeddings, of)

if __name__ == "__main__":
    main()



