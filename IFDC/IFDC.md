## IFDC for Image Difference Captioning

Code for paper "Image Difference Captioning with Instance-Level Fine-Grained Feature Representation", TMM. 

### Prerequisites

- Python 3.7
- PyTorch == 1.4.0

### Data: 

- Dataset
  - CLEVR-Change: Download data from here: [CLEVR-Change Dataset](https://github.com/Seth-Park/RobustChangeCaptioning)
  - Spot-the-Diff: Download data from here: [Spot-the-Diff Dataset](https://github.com/harsh19/spot-the-diff)
- Data preprocessing
  - Build vocab and label files using caption annotations

    - Run following command and obtain the a json file and an hdf5 file in the ``data`` folder. 

    ```py
    python scripts/prepro_labels.py
    ```

  - Visual features and pixel coordinates extraction
    - Extract visual features and pixel coordinates using [Faster R-CNN](https://github.com/peteanderson80/bottom-up-attention). 

  - Semantic attributes extraction

    - Download [CLEVR dataset](https://cs.stanford.edu/people/jcjohns/clevr/) and train [multi-label classifier](https://github.com/pangwong/pytorch-multi-label-classifier) for the experiments on CLEVR-Change dataset.  
    - Download [Pascal VOC dataset](https://pjreddie.com/projects/pascal-voc-dataset-mirror/) and train [multi-label classifier](https://github.com/pangwong/pytorch-multi-label-classifier) for the experiments on Spot-the-Diff dataset. 
    - Run following command and obtain pixel coordinate clustering files of ``change`` and ``nochange``. 

    ```py
    python box.py
    ```

    - Run following command and obtain the input test file of [multi-label classifier](https://github.com/pangwong/pytorch-multi-label-classifier) for ``change`` and ``nochange``. 

    ```python
    python coord.py
    ```

    - Run following command and obtain the input semantic file of IFDC model. 

    ```python
    python coord_transfer1.py
    python coord_transfer2.py
    ```

### Training

- Open ``eval_utils.py`` and find function ``eval_split``, and then set ``split`` to ``val``. 

- Commend out the ``model.eval()`` statement, but keep the ``model = model.module`` statement. 

- For the input to the model, the dimension of visual features is ``[B, nums_oje, 2048]``, the dimension of semantic features is ``[B, nums_oje, 4]``, the dimension of positional features is ``[B, nums_oje, 4]``.  

- Download the ``glove.6B.300d.txt`` file and put it in the ``data/embedding`` folder. Then run following command to get the ``embedding.pt`` file which is used to encode the semantic attributes. 

```python
python embedding.py
```

- Run following command:

```py
python train.py
```

- The ``model.pth``,  ``model-best.pth``, ``infos_up.pkl``, and ``infos_up-best.pkl`` files will be saved in the ``checkpoint`` folder. 	

### Testing

- Open ``eval_utils.py`` and find function ``eval_split``, and then set ``split`` to ``test``. 
- Comment out the ``model = model.module`` statement, but keep the ``model.eval()`` statement. 
- Run following command:

```pytho
python eval.py
```

- The generated sentences will be saved in the ``vis`` folder. 

### Evaluation

Evaluate generated sentences using project [nlg-eval](https://github.com/Maluuba/nlg-eval). 