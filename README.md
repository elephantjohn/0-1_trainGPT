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
   
其中一个变量是self.tok_embeddings = nn.Embedding(params.vocab_size, params.dim)，原理是把所有的输入文字句子按照词表找到索引，即转化为了一个代表句子的向量，再将此向量经过embegging变为例如n*m*3的矩阵，这样，每一个单词都是一个m*3的矩阵，n是总句子数，m是每个句子里的单词数，3是embedding向量的维度；举例：
```python
词汇表是：
单词	索引
“I”	0
“love”	1
“you”	2
“and”	3
“AI”	4
---

input_indices = torch.tensor([
    [0, 1, 4],  # 句子 1 的索引序列
    [2, 3, 4]   # 句子 2 的索引序列
])  # 形状: (batch_size=2, sequence_length=3)

--

output_embeddings = embedding_layer(input_indices)
print(output_embeddings.shape)  # 输出形状: (2, 3, 3)

--

对于句子 1 [0, 1, 4]：
索引 0 对应嵌入向量 [0.1, 0.2, 0.3]（“I” 的嵌入向量）。
索引 1 对应嵌入向量 [-0.1, 0.0, 0.5]（“love” 的嵌入向量）。
索引 4 对应嵌入向量 [-0.6, 0.7, 0.2]（“AI” 的嵌入向量）。 输出为：
[[ 0.1,  0.2,  0.3],
 [-0.1,  0.0,  0.5],
 [-0.6,  0.7,  0.2]]

对于句子 2 [2, 3, 4]：
索引 2 对应嵌入向量 [0.4, -0.2, 0.1]（“you” 的嵌入向量）。
索引 3 对应嵌入向量 [0.3, 0.8, -0.5]（“and” 的嵌入向量）。
索引 4 对应嵌入向量 [-0.6, 0.7, 0.2]（“AI” 的嵌入向量）。 输出为：
[[ 0.4, -0.2,  0.1],
 [ 0.3,  0.8, -0.5],
 [-0.6,  0.7,  0.2]]

最终的输出张量为：

[
    [[ 0.1,  0.2,  0.3],  # 句子 1 的第一个单词 "I"
     [-0.1,  0.0,  0.5],  # 句子 1 的第二个单词 "love"
     [-0.6,  0.7,  0.2]], # 句子 1 的第三个单词 "AI"

    [[ 0.4, -0.2,  0.1],  # 句子 2 的第一个单词 "you"
     [ 0.3,  0.8, -0.5],  # 句子 2 的第二个单词 "and"
     [-0.6,  0.7,  0.2]]  # 句子 2 的第三个单词 "AI"
]


```
