#! /bin/bash

cd /root/work/minimind/ #minimind git top dir
mkdir dataset
cd /root/work/minimind/dataset

# mobvoi_seq_monkey_general_open_corpus.zip
wget -O mobvoi_seq_monkey_general_open_corpus.zip https://huggingface.co/datasets/jingyaogong/minimind_dataset/resolve/main/mobvoi_seq_monkey_general_open_corpus.zip?download=true &

# pretrain_data.csv
wget -O pretrain_data.csv https://huggingface.co/datasets/jingyaogong/minimind_dataset/resolve/main/pretrain_data.csv?download=true & 

# sft_data_multi.csv
wget -O sft_data_multi.csv https://huggingface.co/datasets/jingyaogong/minimind_dataset/resolve/main/sft_data_multi.csv?download=true & 

# sft_data_single.csv
wget -O sft_data_single.csv https://huggingface.co/datasets/jingyaogong/minimind_dataset/resolve/main/sft_data_single.csv?download=true & 

# tokenizer_train.jsonl
wget -O tokenizer_train.jsonl https://huggingface.co/datasets/jingyaogong/minimind_dataset/resolve/main/tokenizer_train.jsonl?download=true &

mkdir dpo
wget -O dpo/dpo_train_data.json https://huggingface.co/datasets/jingyaogong/minimind_dataset/resolve/main/dpo/dpo_train_data.json?download=true & 
wget -O dpo/dpo_zh_demo.json https://huggingface.co/datasets/jingyaogong/minimind_dataset/resolve/main/dpo/dpo_zh_demo.json?download=true &
wget -O dpo/huozi_rlhf_data.json https://huggingface.co/datasets/jingyaogong/minimind_dataset/resolve/main/dpo/huozi_rlhf_data.json?download=true &
wget -O dpo/train_data.json https://huggingface.co/datasets/jingyaogong/minimind_dataset/resolve/main/dpo/train_data.json?download=true &
