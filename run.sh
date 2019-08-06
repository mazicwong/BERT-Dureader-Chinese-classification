export GLUE_DIR=./GLUE_DIR/
export TASK_NAME=QQP
export BERT_DIR=uncased_L-12_H-768_A-12


export GLUE_DIR=./GLUE_DIR/
export TASK_NAME=Dureader
export BERT_DIR=~/pytorch-transformers-test/bert-base-chinese/


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




