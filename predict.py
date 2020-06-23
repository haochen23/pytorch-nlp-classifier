import spacy
import torch
from model.classifier import classifer
from utils import load_data
import config
import dill

nlp=spacy.load("en")

with open("./output/TEXT.Field", "rb") as f:
    TEXT = dill.load(f)

def predict(model, sentence):
    tokenized = [tok.text for tok in nlp.tokenizer(sentence)]  #tokenize the sentence 
    indexed = [TEXT.vocab.stoi[t] for t in tokenized]          #convert to integer sequence
    length = [len(indexed)]                                    #compute no. of words
    tensor = torch.LongTensor(indexed).to(config.device)              #convert to tensor
    tensor = tensor.unsqueeze(1).T                             #reshape in form of batch,no. of words
    length_tensor = torch.LongTensor(length)                   #convert to tensor
    prediction = model(tensor, length_tensor)                  #prediction 
    return prediction.item()

if __name__== "__main__":
    path = "./output/best_model.pt"
    model = torch.load(path)
    model.eval()
    text1 = "Why Chinese girls go crazy about marrying Jack Ma?"
    result = predict(model, text1)
    print("Results close to 1 represent insincere questions.")
    print("Results close to 0 represent sincere questions.")
    print('------')
    print("The result for '{}' is {}".format(text1, result))
    
    text2 = "Why peole spend money on Lottery?"
    result = predict(model, text2)
    print("The result for '{}' is {}".format(text2, result))