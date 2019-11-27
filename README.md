# nli-lstm-bert
This is a repo for natural language inference using Match-LSTM and pre-trained BERT. This repo is based on https://github.com/donghyeonk/match-lstm.


## Examples
SNLI 1.0 dataset will be automatically downloaded to data/ and unzipped if it doesn't exist.


Train the model with default parameters.
```
python main.py
```

Train the model on GPU 0 and 1.

```
python main.py --gpu=0,1
```

Train the model on GPU 0 and finetune Bert.

```
python main.py --gpu=0 --train_bert
```

Recommended training command:

```
python main.py --batch_size=512 --gpu=0
```

If you want to fine-tune Bert which increases the number of parameters significantly, you'd better reduce the batch size.
```
python main.py --batch_size=32 --gpu=0 --train_bert
or 
python main.py --batch_size=64 --gpu=0,1 --train_bert
```