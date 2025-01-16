# 0-1_trainGPT
### env环境配置
CPU: Intel(R) Xeon(R) Gold 6133 CPU @ 2.50GHz  或  Intel(R) Xeon(R) Platinum 8352V CPU @ 2.10GHz
内存：28 GB 或 224G
显卡：NVIDIA GeForce RTX 3090(24GB) * 2 或 4090*8
环境：python 3.11.0 + Torch 2.5.1 + DDP单机多卡训练 

### 0. 下载minimind仓库
将https://github.com/elephantjohn/minimind 下载到 /root/work/目录下；

下载后，/root/work/minimind目录下为minimind项目代码

### 1.下载数据集
##### 1.1 下载 tokenizer训练集和 Pretrain数据集
```bash
cp download_dataset_tokenizer.sh /root/work/minimind/
cd /root/work/minimind
bash download_dataset_tokenizer.sh
```
执行后，将新建dataset目录，tokenizer训练集和 Pretrain数据集会被下载到/root/work/minimind/的dataset目录下
##### 1.2 下载SFT数据集 (预训练阶段暂时用不到)
```
cd  /root/work/minimind/dataset
wget  -O sft_data_zh.jsonl www.modelscope.cn/datasets/deepctrl/deepctrl-sft-data/resolve/master/sft_data_zh.jsonl

```
### 2.预训练
在 RTX 3090(24GB) * 2上进行预训练
```python
cd /root/work/minimind/
torchrun --nproc_per_node 2 1-pretrain.py
```
LLM总参数量：26.878 million
预训练中途打印：

<img width="758" alt="截屏2025-01-15 21 04 29" src="https://github.com/user-attachments/assets/44f59c4a-ee48-4903-9727-78c225b05fa1" />

预训练资源占用：

<img width="755" alt="截屏2025-01-15 21 06 22" src="https://github.com/user-attachments/assets/3ba51cd1-6430-45e1-ad4a-6eae02ddb629" />

###  代码思路
1. 　从huggingface的transformers库引入了 AutoTokenizer 和 AutoModelForCausalLM；AutoTokenizer 是一个通用的分词器加载类；

2.  通过 tokenizer = AutoTokenizer.from_pretrained('./model/minimind_tokenizer') 加载一个tokenizer分词器对象，准备在预训练时使用；
  './model/minimind_tokenizer'下有四个文件，vocab.json、merges.txt、tokenizer.json，tokenizer_config.json，他们可以通过Hugging Face Tokenizers 库训练一个 BPE 分词器得到，训练完成后，分词器会生这些文件；
具体训练过程可参考
```python
from tokenizers import Tokenizer, models, trainers
# 初始化一个空的 BPE 分词器
tokenizer = Tokenizer(models.BPE())
# 定义训练器
trainer = trainers.BpeTrainer(vocab_size=30000, special_tokens=["[PAD]", "[CLS]", "[SEP]"])
# 在文本数据上训练分词器
tokenizer.train(["data.txt"], trainer)
# 保存分词器
tokenizer.save("./model/minimind_tokenizer/tokenizer.json")
```


3. 关于自定义的Transformer类
   - embedding
其中一个变量是self.tok_embeddings = nn.Embedding(params.vocab_size, params.dim)，原理是把所有的输入文字句子按照词表找到索引，即转化为了一个代表句子的向量，再将此向量经过embegging变为例如n*m*3的矩阵，这样，每一个单词都是一个m*3的矩阵，n是总句子数，m是每个句子里的单词数，3是embedding向量的维度；举例：
```python
词汇表是：
单词	索引
“I”	0
“love”	1
“you”	2
“and”	3
“AI”	4
```

嵌入层的初始化：
```python
import torch
import torch.nn as nn

# 初始化嵌入层，词汇表大小为 5，嵌入向量维度为 3
embedding_layer = nn.Embedding(num_embeddings=5, embedding_dim=3)
这一行定义了一个嵌入层（embedding layer），用于将输入的词（token）从离散的词汇表索引（vocab_size）映射到一个连续的向量空间（dim）。

params.vocab_size 是词汇表的大小（即有多少个不同的词）。
params.dim 是嵌入向量的维度（每个词被表示为一个 dim 维的向量）。
嵌入矩阵的形状是 (vocab_size, dim)，每一行表示一个词的嵌入向量。
这一步的作用是将离散的词索引（如 [0, 1, 2]）映射为连续的嵌入向量（如 [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]）。

```

假设我们有两个句子：
```python

句子 1: "I love AI" -> 索引序列 [0, 1, 4]
句子 2: "you and AI" -> 索引序列 [2, 3, 4]
我们将这两个句子组成一个批次（batch），输入张量的形状为 (batch_size, sequence_length)，即：

input_indices = torch.tensor([
    [0, 1, 4],  # 句子 1 的索引序列
    [2, 3, 4]   # 句子 2 的索引序列
])  # 形状: (batch_size=2, sequence_length=3)

```

将输入张量传入嵌入层：
```python
output_embeddings = embedding_layer(input_indices)
print(output_embeddings.shape)  # 输出形状: (2, 3, 3)
```

输出的形状为 (batch_size, sequence_length, embedding_dim)，即：
```python
batch_size=2：表示有 2 个句子。
sequence_length=3：每个句子有 3 个单词。
embedding_dim=3：每个单词的嵌入向量是 3 维的。
```

对于句子 1 [0, 1, 4]：
```python
索引 0 对应嵌入向量 [0.1, 0.2, 0.3]（“I” 的嵌入向量）。
索引 1 对应嵌入向量 [-0.1, 0.0, 0.5]（“love” 的嵌入向量）。
索引 4 对应嵌入向量 [-0.6, 0.7, 0.2]（“AI” 的嵌入向量）。 输出为：
[[ 0.1,  0.2,  0.3],
 [-0.1,  0.0,  0.5],
 [-0.6,  0.7,  0.2]]
```

对于句子 2 [2, 3, 4]：
```python
索引 2 对应嵌入向量 [0.4, -0.2, 0.1]（“you” 的嵌入向量）。
索引 3 对应嵌入向量 [0.3, 0.8, -0.5]（“and” 的嵌入向量）。
索引 4 对应嵌入向量 [-0.6, 0.7, 0.2]（“AI” 的嵌入向量）。 输出为：
[[ 0.4, -0.2,  0.1],
 [ 0.3,  0.8, -0.5],
 [-0.6,  0.7,  0.2]]
```

最终的输出张量为：
```python

[
    [[ 0.1,  0.2,  0.3],  # 句子 1 的第一个单词 "I"
     [-0.1,  0.0,  0.5],  # 句子 1 的第二个单词 "love"
     [-0.6,  0.7,  0.2]], # 句子 1 的第三个单词 "AI"

    [[ 0.4, -0.2,  0.1],  # 句子 2 的第一个单词 "you"
     [ 0.3,  0.8, -0.5],  # 句子 2 的第二个单词 "and"
     [-0.6,  0.7,  0.2]]  # 句子 2 的第三个单词 "AI"
]
```

nn.Embedding 的嵌入矩阵是一个可训练的参数，存储在 embedding_layer.weight 中。你可以通过以下方式查看嵌入矩阵的初始值：
```python
print(embedding_layer.weight)
结果：

tensor([[ 0.1234, -0.5678,  0.9101],
        [-0.2345,  0.6789, -0.1011],
        [ 0.3456, -0.7890,  0.1213],
        [-0.4567,  0.8901, -0.1415],
        [ 0.5678, -0.9012,  0.1617]], requires_grad=True)
```

嵌入矩阵的初始化发生在 创建 nn.Embedding 对象时，也就是执行 embedding_layer = nn.Embedding(...) 的那一刻。

如果你没有手动修改嵌入矩阵的值，那么它会一直保持默认的随机初始化值，直到训练过程中被更新（通过反向传播）。

嵌入矩阵是一个可训练的参数（requires_grad=True），在训练过程中会通过反向传播自动更新。每次梯度更新时，嵌入矩阵的值都会被调整，以更好地表示输入数据的语义关系。

- dropout

  nn.Dropout是PyTorch中的一个层，用于在训练过程中随机将一部分神经元的输出设为零，以防止过拟合;

  params.dropout是一个参数，指定了在训练过程中每个神经元被“丢弃”的概率。这个值通常在0到1之间。

```python
  self.dropout = nn.Dropout(params.dropout)
```

- layers

torch.nn.ModuleList是一个特殊的容器，用于存储一系列的子模块（例如，神经网络层）。

使用ModuleList可以方便地管理和迭代多个层或模块，特别是在需要动态添加或修改网络结构时。

  ```python
  self.layers = torch.nn.ModuleList():
  ```

```python
  self.dropout = nn.Dropout(params.dropout)
  self.layers = torch.nn.ModuleList()
```
以上这两行代码是在定义一个神经网络的部分结构，其中包括一个Dropout层和一个用于存储其他层的列表。

```python
   for layer_id in range(self.n_layers):
       self.layers.append(TransformerBlock(layer_id, params))
```
在每一层创建一个新的TransformerBlock实例，并将其添加到self.layers列表中。




