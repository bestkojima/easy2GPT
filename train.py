



import dateset



# 参数设置


lr=5e-5
batch_size=4
epochs=10
max_len=512
warmup_steps=1000
gradient_accumulation_steps=1
weight_decay=0.01
adam_epsilon=1e-8
logging_steps=10
eval_steps=100
save_steps=100
seed=42
fp16=False
fp16_opt_level='O1'
model_name_or_path='gpt2'
output_dir='output'
data_path='data'
do_train=True
do_eval=True
do_predict=True
per_device_train_batch_size=4
per_device_eval_batch_size=4

from dataload import get_dataloader
import torch.nn.functional as F
import torch
import torch.nn as nn

dataloader=get_dataloader()


def calu_batch_loss(model, input,target):
    input,target=torch.tensor(input),torch.tensor(target)
    return F.cross_entropy(model(input).flatten(), target.flatten())

def train(epochs,dataloader,model,optimizer,):
    train_losses, val_losses, track_tokens_seen = [], [], []
    

    # 添加tqdm
    for epoch in range(epochs):
        model.train()
    # 训练
    
        for i,(input_batch,target_batch) in enumerate(dataloader):
           optimizer.zero_grad()
           loss=calu_batch_loss(model, input_batch,target_batch)
           loss.backward()
           optimizer.step()
           train_losses.append(loss.item())
           

           #TODO 添加验证 和保存
           
           
        