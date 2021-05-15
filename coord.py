import json
from tqdm import tqdm

inputfile = '/home/ubuntu/LY/self-critical.pytorch-master/data/label/label1_25_4+4.json'
outputfile = '/home/ubuntu/LY/self-critical.pytorch-master/data/label/coord1_25_4+4.json'

input = open(inputfile, "r", encoding='utf-8')
output = open(outputfile, 'w', encoding='utf-8')

for line in tqdm(input):
    line = json.loads(line.strip())
    image_id = line.get('id')
    coord = line.get('coord')
    if image_id <= 39999:
        image_file = "/home/ubuntu/LY/dataset/sc_images/CLEVR_semantic_" + str(image_id).zfill(6) + ".png"
        # image_file = "/home/ubuntu/LY/dataset/images/CLEVR_default_" + str(image_id).zfill(6) + ".png"
    else:
        image_file = "/home/ubuntu/LY/dataset/nsc_images/CLEVR_nonsemantic_" + str(image_id-40000).zfill(6) + ".png"
        # image_file = "/home/ubuntu/LY/dataset/images/CLEVR_default_" + str(image_id-40000).zfill(6) + ".png"
    for k, v in coord.items():
        num_k = k
        x, y = v[0], v[1]
        w, h = 105, 105
        example = {"box": {"y": y, "x": x, "w": w, "h": h}, "image_id": image_id, "num": num_k, "image_file": image_file, "size": {"width": 480, "height": 320}}
        output.write(json.dumps(example) + '\n')