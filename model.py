import torch
import torch.nn as nn
import numpy as np
config={
    "vocab_size": 50257,
    "block_size": 1024, # seq length
    "n_layer": 24,
    "n_head": 16,
    "n_embd": 16,
    "dropout": 0.1,
    "bias": True
}

class mlp(nn.Module):
    def __init__(self, input_size, hidden_size, output_size,dropout=0.2):
        super(mlp, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
    
        self.layers=nn.Sequential(
            nn.Linear(input_size, 4*hidden_size),
            nn.GELU(),
            nn.Linear(4*hidden_size, output_size),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.layers(x)

class multiHeadAttention(nn.Module):
    def __init__(self, input_size=config["n_embd"],hidden_size=config["n_embd"],out_size=config["n_embd"], n_head=config["n_head"], dropout=0.2):
        super(multiHeadAttention, self).__init__()

        """
        Args:
            input_size: vocal_size
            
            
        """
        assert hidden_size % n_head == 0
        self.n_embd = hidden_size
        self.n_head = hidden_size
        self.head_size = hidden_size // n_head


        self.qkv=nn.Linear(input_size, 3 * hidden_size)


        self.softmax=nn.Softmax(-1)
        self.dropout=nn.Dropout(dropout)
        self.out=nn.Linear(hidden_size, out_size)

        
    
    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        q,k,v=self.qkv(x).chunk(3,dim=-1)
        
        q=q.view(batch_size,seq_len,self.n_head,self.head_size).permute(0,2,1,3)
        k=k.view(batch_size,seq_len,self.n_head,self.head_size).permute(0,2,3,1)
        v=v.view(batch_size,seq_len,self.n_head,self.head_size).permute(0,2,1,3)


        attn_scores=q@k/self.head_size**0.5

        mask=torch.full_like(attn_scores,float("-inf")).triu(1)
        
        attn_scores+=mask
        
        attn_weight=self.softmax(attn_scores)
        
        attn_weight=self.dropout(attn_weight)
        
        out=attn_weight@v
        
        out=out.permute(0,2,1,3).contiguous().view(batch_size,seq_len,self.n_embd)
        return self.out(out)


class transformerblock(nn.Module):
    def __init__(self, input_size=config["n_embd"],hidden_size=config["n_embd"],out_size=config["n_embd"], n_head=config["n_head"], dropout=0.2):
        super(transformerblock, self).__init__()
        
        self.ln1=nn.LayerNorm(input_size)
        self.mha=multiHeadAttention()
        self.dropout1=nn.Dropout(dropout)
       
        self.ln2=nn.LayerNorm(hidden_size)
        self.mlp=mlp(hidden_size,hidden_size,hidden_size,dropout)
        self.dropout2=nn.Dropout(dropout)



    def forward(self, x):
        x=x+self.dropout1(self.mha(self.ln1(x)))
        x=x+self.dropout2(self.mlp(self.ln2(x)))
        return x


###############  tokenizer ##################


# 假设 已经转成了token
class embAndpos(nn.Module):
    def __init__(self, vocab_size=config["vocab_size"], block_size=config["block_size"], n_embd=config["n_embd"]):
        super(embAndpos, self).__init__()
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.n_embd = n_embd
        self.token_emb=nn.Embedding(vocab_size, n_embd)
        self.token_pos=nn.Embedding(block_size, n_embd)
    def forward(self, x):
        batch_size, seq_len = x.size()
        token_emb=self.token_emb(x)
        token_pos=self.token_pos(torch.arange(self.block_size))[:seq_len]
        
        return token_emb+token_pos

class GPT2(nn.Module):
    def __init__(self, vocab_size=config["vocab_size"], block_size=config["block_size"], n_embd=config["n_embd"], n_head=config["n_head"], n_layer=config["n_layer"], dropout=0.2):
        super(GPT2, self).__init__()


        # 输入为 batch seq_len token形式
        self.embAndpos=embAndpos(vocab_size, block_size, n_embd)
        self.transformer=nn.Sequential(
            *[transformerblock(n_embd, n_embd, n_embd, n_head, dropout) for _ in range(n_layer)]
        )
        self.ln_f=nn.LayerNorm(n_embd)
        self.lm_head=nn.Linear(n_embd, vocab_size)


    def forward(self, x):
        x=self.embAndpos(x)
        x=self.transformer(x)
        x=self.ln_f(x)
        logits=self.lm_head(x)
        
        return logits



def generate(index,max_new_token,cxt_size,model):
    
    """
    :param index:  输入的index
    :param max_new_token: 生成token的个数
    :param cxt_size: 上下文长度
    :param model: 模型
    
    """

    for _ in range(max_new_token):
        
        token_series=index[:,-cxt_size:]
        print(token_series.shape)
        logits=model(token_series)
        print(logits.shape)
        logits=logits[:, -1, :]
        print(logits.shape)

        logits=torch.argmax(logits, dim=-1) 
        print(logits.shape)


        index=torch.cat([index, logits.unsqueeze(1)], dim=-1)
        print(index.shape)
        
    return index
        



        
if __name__ == "__main__":
    # i=torch.randn(4,5,config["n_embd"]) #already pos&emb
    i=torch.randint(0,20,(4,5)) # str2index
    #print(i.shape)
    #TODO tokenizer未实现
    c=GPT2()
    #print(c(i).size())
    generate(i,2,3,c)
   
    # 输出 为 index  需要 index2str

        