import config
from utils import load_data, binary_accuracy
from model.classifier import classifer
import torch 
from torchtext import data
import torch.optim as optim


# count trainable parameters
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train(model, iterator, optimizer, criterion):

    # initialize every epoch
    epoch_loss = 0
    epoch_acc = 0

    # set the model in training mode
    model.train()
    
    for batch in iterator:
        #reset the gradients after every batch
        optimizer.zero_grad()

        # retrieve text and num of words
        text, text_lengths = batch.text

        # convert to 1D tensor
        predictions = model(text, text_lengths).squeeze()

        # compute the loss
        loss = criterion(predictions, batch.label)
        # print(loss.item())
        # compute binary accuracy
        acc = binary_accuracy(predictions, batch.label)
        # print(acc.item())

        # backpropogate the loss and compute the gradients
        loss.backward()

        # update weights
        optimizer.step()

        # loss and accuracy
        epoch_loss += loss.item()
        epoch_acc += acc.item()
    return epoch_loss/len(iterator), epoch_acc/len(iterator)

def evaluate(model, iterator, criterion):

    # initialize every epoch
    epoch_loss = 0
    epoch_acc = 0

    #set model to eval mode
    model.eval()

    # deactivates auto_grad
    with torch.no_grad():
        for batch in iterator:
            # retrieve text and num of words
            text, text_lengths = batch.text 
            
            # convert to 1D tensor
            predictions = model(text, text_lengths).squeeze()

            loss = criterion(predictions, batch.label)
            acc = binary_accuracy(predictions, batch.label)

            # keep track of loss and accuracy
            epoch_loss += loss.item()
            epoch_acc += acc.item()
    
    return epoch_loss/len(iterator), epoch_acc/len(iterator)

if __name__ == "__main__":
    train_iterator, valid_iterator, TEXT, LABEL = load_data("./data/quora_labeled.csv")
    # define hyperparameters
    size_of_vocab = len(TEXT.vocab)
    embedding_dim = 100
    num_hidden_nodes = 32
    num_output_nodes = 1
    num_layers = 2
    bidirectional = True
    dropout = 0.2
    device = config.device

    model = classifer(size_of_vocab, 
                      embedding_dim, 
                      num_hidden_nodes, 
                      num_output_nodes,
                      num_layers,
                      bidirectional,
                      dropout)
    print(model)
    print(f'The model has {count_parameters(model):,} trainable parameters')
    
    # Initialize the pretrained embedding
    pretrained_embeddings = TEXT.vocab.vectors
    model.embedding.weight.data.copy_(pretrained_embeddings)
    #define optimizer and loss
    optimizer = optim.Adam(model.parameters())
    criterion = nn.BCELoss()
    
    # push to cuda if available
    model = model.to(device)
    criterion = criterion.to(device)
    N_EPOCHS = config.N_EPOCH
    best_valid_loss = float('inf')
    
    # start training
    for epoch in range(N_EPOCHS):

        # train the model
        train_loss, train_acc = train(model, train_iterator, optimizer, criterion)

        # evaluate the model
        valid_loss, valid_acc = evaluate(model, valid_iterator, criterion)

        # save the best model
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            # torch.save(model.state_dict(), './output/best_model.pt')
            torch.save(model, './output/best_model.pt')
        print("Epoch {}/{}:".format(epoch+1, N_EPOCHS))
        print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')
