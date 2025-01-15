# 0-1_trainGPT
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
在 2 x 3090上进行预训练
```python
cd /root/work/minimind/dataset
torchrun --nproc_per_node 2 1-pretrain.py
```
训练过程：

<img width="758" alt="截屏2025-01-15 21 04 29" src="https://github.com/user-attachments/assets/44f59c4a-ee48-4903-9727-78c225b05fa1" />



