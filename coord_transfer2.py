import json
from tqdm import tqdm
import numpy as np
label_file = '/home/ubuntu/LY/self-critical.pytorch-master/data/label/label_25_4+1.json'
default_file = "/home/ubuntu/LY/self-critical.pytorch-master/data/label/default_25_4+1_change.json"
output_file = "/home/ubuntu/LY/self-critical.pytorch-master/data/outfile/4+1/output.json"
# output_file = "D:/myfile/codes/self-critical.pytorch-master/data/output.json"
default_outfile = "/home/ubuntu/LY/self-critical.pytorch-master/data/label/default_25_4+1_out.json"
of = open(output_file, 'r', encoding="utf-8")
unit = json.load(of)
ixtow = unit["ix_to_word"]
wtoix = {w: ix for ix, w in ixtow.items()}
# print(wtoix)

lf = open(label_file, 'r', encoding='utf-8')
df = open(default_file, 'r', encoding='utf-8')
df_op = open(default_outfile, 'w', encoding='utf-8')

unit = json.load(df)

semantic_dic = dict()
n = 0
for line in tqdm(lf):
    n += 1
    line = json.loads(line.strip())
    image_id = line.get('id')
    clu = line.get("clu")
    att_ix = []
    for k, v in clu.items():
        num = int(k)
        ix = []
        i = str(image_id) + "_" + str(num)
        attibute = unit[i]
        for w in attibute:
            ix.append(wtoix[w])
        for i in range(len(v)):
            att_ix += [ix]
    semantic_dic[image_id] = att_ix
df_op.write(json.dumps(semantic_dic))

