import torch
import numpy as np
import torch.nn.functional as F
import os
from tqdm import tqdm
import json

file = ""
change = ""

f = open(file, 'r', encoding='utf-8')
c = open(change, 'w', encoding='utf-8')
d = dict()

for line in tqdm(f):
    line = json.loads(line.strip())
    att = line.get("attibute")
    id = line.get("image_id")
    num = line.get("num")
    i = str(id) + "_" + str(num)
    d[i] = att
c.write(json.dumps(d))
