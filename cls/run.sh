model='unixcoder-base'
ifraw='False'
input_prompt='cin>>'
output_prompt='cout<<'
input_prompt_sep=';'
n_class=104

CUDA_VISIBLE_DEVICES=0,1 nohup python run.py \
                --output_dir=./savedmodels \
                --model_type=roberta_cls \
                --config_name=../microsoft/$model \
                --model_name_or_path=../microsoft/$model \
                --tokenizer_name=../microsoft/$model \
                --do_train \
                --do_test \
                --train_data_file=../dataset/train_fuzz_cls.jsonl \
                --eval_data_file=../dataset/valid_fuzz_cls.jsonl \
                --test_data_file=../dataset/test_fuzz_cls.jsonl \
                --epoch 10 \
                --block_size 512 \
                --train_batch_size 32 \
                --eval_batch_size 64 \
                --learning_rate 2e-5 \
                --max_grad_norm 1.0 \
                --evaluate_during_training \
                --input_prompt "$input_prompt" \
                --output_prompt "$output_prompt" \
                --input_prompt_sep $input_prompt_sep \
                --need_raw $ifraw \
                --nohang \
                --nocrash \
                --n_class $n_class \
                --seed 123456 2>&1| tee ./log.txt &
