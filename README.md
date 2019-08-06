# BERT-Dureader-Chinese-classification
使用BERT模型做Dureader2.0数据集中的分类任务

### 前置库
- pytorch-transformers
- tensorboardX
- scikit-learn

### 数据集预处理
Dureader2.0是一个阅读理解数据集, 问题被分为[`Description`, `Entity`, `Yes/No`], 将其中的`Yes/No`问题拿出来, 单独作为分类数据集训练,

数据集可以从这里获取[https://ai.baidu.com/broad/subordinate?dataset=dureader](https://ai.baidu.com/broad/subordinate?dataset=dureader)

我写了个转换文件`json2tsv.py`, 将原始数据集转为run_glue.py需要的格式,

转换后数据集分布如下 共有四类标签:
```
label = ["Yes", "No", "Depends", "No_Opinion"]
train = [8486,  5222,  3149,      203]  = 17060
dev   = [242,   208,   0,         0]    = 450
```

标签有四类: label2idx = {"Yes":'0', "No":'1', "Depends":'2', "No_Opinion":'3'}

### 步骤
- 1.把整个[pytorch-transformer](https://github.com/huggingface/pytorch-transformers)下载下来, 需要修改的文件只有两个, utils_glue.py 和 run_glue.py
- 2.准备预训练模型, 参考这里[converting_tensorflow_models.py](https://huggingface.co/pytorch-transformers/converting_tensorflow_models.html#bert)
- 3.修改utils_glue.py的过程, 可以全局搜索下`QQP`, 即quora文本分类的数据集, 然后`QQP`出现的地方, 都复制多一份, 改成`Dureader`就行
- 4.修改run_glue.py的过程, 是为了输出具体的文本标签, 在`def evaluate()`中把`tmp_eval_loss, logits = outputs[:2]`的logits输出就可以


### 训练及预测脚本
```
export GLUE_DIR=./GLUE_DIR/
export TASK_NAME=Dureader
export BERT_DIR=~/bert-base-chinese/

python ./run_glue.py \
    --model_type bert \
    --model_name_or_path $BERT_DIR  \
    --task_name $TASK_NAME \
    --do_train \
    --do_eval \
    --do_lower_case \
    --data_dir $GLUE_DIR/$TASK_NAME \
    --max_seq_length 128 \
    --per_gpu_eval_batch_size=8   \
    --per_gpu_train_batch_size=8   \
    --learning_rate 2e-5 \
    --num_train_epochs 3.0 \
    --output_dir ./tmp/$TASK_NAME/
```

### 结果
global_step = 1068

average loss = 0.783466279422969

acc = 0.6822222222222222

### 注意事项
- 中英文的区别只是pretrain model的vocab.txt的区别
- 注意有个坑点, 步骤二中预训练模型转换, 记得把转换后的bert_config.json修改为config.json


# 最后
本repo是对Dureader2.0数据集训练模型的一个补充, 也是一个独立的分类模型, 之前在Dureader中对span extract任务和classification任务做multi-task learning, 分类这块效果不是很好, 所以改用bert

在测试集测试后发现, 加入bert训练的分类模型, 比原模型提升了0.5的ROUGE和BLEU

|Model|ROUGE-L|BLEU-4|
| -- | -- | -- |
|**Before:** BERT-span|51.49|49.7|
|**After:** BERT-span + BERT-classification|51.93 |50.11|

