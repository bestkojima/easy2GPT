import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import tiktoken
from pprint import pp,pprint

    

class GPT2Dataset(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):

        self.tokenizer = tokenizer
        self.input_index = []
        self.target_index = []

        self.token=self.tokenizer.encode(txt, allowed_special={'<|endoftext|>'})

        for i in range(0,len(self.token)-max_length,stride):
            self.input_index.append(torch.tensor(self.token[i:i+max_length]))
            self.target_index.append(torch.tensor(self.token[i+1:i+max_length+1]))
            
    def __len__(self):
        return len(self.input_index)
    

    def __getitem__(self, index) :
        return self.input_index[index], self.target_index[index]
    


def get_dataloader(txt,max_length=256,stride=128,batch_size=4,
                   shuffle=True,drop_last=True):
    tokenizer=tiktoken.get_encoding("gpt2")
    

    dataset=GPT2Dataset(txt,tokenizer,max_length,stride)
    print((len(dataset)))
    dataloader=DataLoader(
        dataset,batch_size=batch_size,shuffle=shuffle,drop_last=drop_last)
    return dataloader


if __name__=="__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with open(file="test.txt") as f:
        txt=f.read()
    dataloader=get_dataloader(txt,batch_size=2, max_length=256, stride=256)
    vocab_size = 50257
    output_dim = 256
    c=nn.Embedding(vocab_size,output_dim)
    for batch in dataloader:
        x, y = batch
        print(c(x).shape,y.shape)
        

    


    
    
    