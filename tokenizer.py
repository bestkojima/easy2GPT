# 默认输入语言为英文 ，因此不需要 类似 jieba分词 如 heelo， row[emoji]，可经过正则 表达式处理
# raw text-> 正则表达式 -> 单词表 因此不需要分词
from collections import defaultdict


# 假设输入的都是干净英文
class BPE():
    def __init__(self, vocab_size=30000):
        self.vocab_size=vocab_size
        self.str2index={}
        self.index2str={}


    def train(self,corups):
        


        self.str2index={char:index for index ,char in enumerate(set("".join(corups)))}
        self.index2str={index:char for index ,char in enumerate(set("".join(corups)))}

        while len(self.vocab)<self.vocab_size:
            pairs=self._get_pairs(corups)
        pass




    def _get_pairs(self, corups):
        pairs=defaultdict(int)
        for word in corups:
            for i in range(len(word)-1):
                pairs[word[i], word[i+1]]+=1
        return pairs
    
    
    def _merrege_vocab(self, vocab, pairs):
        new_pairs=[]
        best_pair=max(pairs, key=pairs.get)
        
            
    # encode

    def encode():
        pass

    def decode():
        pass    
        