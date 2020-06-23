import spacy
import torch
from model.classifier import classifer

nlp=spacy.load("en")

def predict(model, sentence):
    tokenized = [tok.text for tok in nlp.tokenizer(sentence)]  #tokenize the sentence 
    indexed = [TEXT.vocab.stoi[t] for t in tokenized]          #convert to integer sequence
    length = [len(indexed)]                                    #compute no. of words
    tensor = torch.LongTensor(indexed).to(device)              #convert to tensor
    tensor = tensor.unsqueeze(1).T                             #reshape in form of batch,no. of words
    length_tensor = torch.LongTensor(length)                   #convert to tensor
    prediction = model(tensor, length_tensor)                  #prediction 
    return prediction.item()

if __name__== "__main__":
    path = "./output/best_model.pt"
    model = torch.load(path)
    model.eval()
    text1 = "Why Indian girls go crazy about marrying Shri. Rahul Gandhi ji?"
    result = predict(model, text1)
    print("The result for {} is {}".format(text1, result))