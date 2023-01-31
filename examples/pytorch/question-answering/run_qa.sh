
# # direct fine-tuning
# python run_qa.py \
#   --model_name_or_path bert-base-uncased \
#   --dataset_name squad \
#   --do_train \
#   --do_eval \
#   --fp16 \
#   --per_device_train_batch_size 12 \
#   --learning_rate 3e-5 \
#   --num_train_epochs 2 \
#   --max_seq_length 384 \
#   --doc_stride 128 \
#   --output_dir tmp/debug_squad0

# # load enhanced base
# python run_qa.py \
#   --model_name_or_path ../language-modeling/tmp/mlm-2/ \
#   --dataset_name squad \
#   --do_train \
#   --do_eval \
#   --fp16 \
#   --per_device_train_batch_size 12 \
#   --learning_rate 3e-5 \
#   --num_train_epochs 2 \
#   --max_seq_length 384 \
#   --doc_stride 128 \
#   --output_dir tmp/debug_squad1


# # new adapter base
# python run_qa.py \
#   --model_name_or_path bert-base-uncased \
#   --dataset_name squad \
#   --train_adapter \
#   --adapter_config houlsby \
#   --do_train \
#   --do_eval \
#   --fp16 \
#   --per_device_train_batch_size 12 \
#   --learning_rate 3e-5 \
#   --num_train_epochs 2 \
#   --max_seq_length 384 \
#   --doc_stride 128 \
#   --output_dir tmp/adapter0


# load adapter base
python run_qa.py \
  --model_name_or_path bert-base-uncased \
  --dataset_name squad \
  --train_adapter \
  --adapter_config houlsby \
  --load_adapter ../language-modeling/tmp/mlm-3/squad \
  --do_train \
  --do_eval \
  --fp16 \
  --per_device_train_batch_size 12 \
  --learning_rate 3e-5 \
  --num_train_epochs 2 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --output_dir tmp/adapter1