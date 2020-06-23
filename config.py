import torch

seed = 42
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True

BATCH_SIZE = 64
device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu:0")
N_EPOCH = 5