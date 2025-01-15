# 0-1_trainGPT
### 0. 下载minimind仓库
将https://github.com/elephantjohn/minimind 下载到 /root/work/目录下
下载后，/root/work/minimind目录下为minimind项目代码

### 1.下载数据集
##### 1.1 下载 tokenizer训练集和 Pretrain数据集
```bash
cp download_dataset_tokenizer.sh /root/work/minimind/
cd /root/work/minimind
bash download_dataset_tokenizer.sh
```
执行后，将新建dataset目录，tokenizer训练集和 Pretrain数据集会被下载到/root/work/minimind/的dataset目录下
##### 1.2 下载其他数据集

