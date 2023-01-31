pt_column_name="context"
num_train_epochs=1

# # mlm-1: no pre-training
# python run_mlm.py \
#     --model_name_or_path bert-base-uncased \
#     --dataset_name squad \
#     --pt_column_name $pt_column_name \
#     --fp16 \
#     --per_device_eval_batch_size 8 \
#     --do_eval \
#     --output_dir tmp/mlm-1


# mlm-2: enhance pre-training
# python run_mlm.py \
#     --model_name_or_path bert-base-uncased \
#     --dataset_name squad \
#     --pt_column_name $pt_column_name \
#     --fp16 \
#     --num_train_epochs $num_train_epochs \
#     --per_device_train_batch_size 8 \
#     --per_device_eval_batch_size 8 \
#     --do_train \
#     --do_eval \
#     --output_dir tmp/mlm-2

# mlm-3: adapter pre-training
python run_mlm.py \
    --model_name_or_path bert-base-uncased \
    --train_adapter \
    --adapter_config houlsby \
    --dataset_name squad \
    --pt_column_name $pt_column_name \
    --fp16 \
    --num_train_epochs $num_train_epochs \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --do_train \
    --do_eval \
    --output_dir tmp/mlm-3

