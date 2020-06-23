# PyTorch NLP Sincereness Detector

## Usage:

__Train the model__
```python
python3 train.py
```

__Inference__
```python
python3 predict.py --question "Why people like football?"

Results close to 1 represent insincere questions.
Results close to 0 represent sincere questions.
------
The result for 'Why people like spend money on Lottery?' is 0.4414695203304291

```
