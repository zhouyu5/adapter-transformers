model_name_or_path='../language-modeling/tmp/unique-10epoch/'
# model_name_or_path='../language-modeling/tmp/mlm-3/squad'
num_train_epochs=1


# direct fine-tuning profiling
python run_qa.py \
  --model_name_or_path bert-base-uncased \
  --dataset_name squad \
  --do_train \
  --do_eval \
  --fp16 \
  --per_device_train_batch_size 12 \
  --learning_rate 3e-5 \
  --num_train_epochs $num_train_epochs \
  --max_seq_length 384 \
  --doc_stride 128 \
  --max_train_samples 100 \
  --max_eval_samples 100 \
  --overwrite_output_dir \
  --output_dir tmp/direct_ft1



# # direct fine-tuning
# python run_qa.py \
#   --model_name_or_path bert-base-uncased \
#   --dataset_name squad \
#   --do_train \
#   --do_eval \
#   --fp16 \
#   --per_device_train_batch_size 12 \
#   --learning_rate 3e-5 \
#   --num_train_epochs $num_train_epochs \
#   --max_seq_length 384 \
#   --doc_stride 128 \
#   --output_dir tmp/direct_ft1

# load enhanced base
# python run_qa.py \
#   --model_name_or_path $model_name_or_path \
#   --dataset_name squad \
#   --do_train \
#   --do_eval \
#   --fp16 \
#   --per_device_train_batch_size 12 \
#   --learning_rate 3e-5 \
#   --num_train_epochs $num_train_epochs \
#   --max_seq_length 384 \
#   --doc_stride 128 \
#   --output_dir tmp/qa-2-3epoch


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
#   --num_train_epochs $num_train_epochs \
#   --max_seq_length 384 \
#   --doc_stride 128 \
#   --output_dir tmp/adapter


# # load adapter base
# python run_qa.py \
#   --model_name_or_path bert-base-uncased \
#   --dataset_name squad \
#   --train_adapter \
#   --adapter_config houlsby \
#   --load_adapter $model_name_or_path \
#   --do_train \
#   --do_eval \
#   --fp16 \
#   --per_device_train_batch_size 12 \
#   --learning_rate 3e-5 \
#   --num_train_epochs $num_train_epochs \
#   --max_seq_length 384 \
#   --doc_stride 128 \
#   --output_dir tmp/adapter1