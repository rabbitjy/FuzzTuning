model='codebert-base'
ifraw='False'
input_prompt='cin>>'
output_prompt='cout<<'
input_prompt_sep=';'
n_candidate=499

# chage CUDA_VISIBLE_DEVICES when needed
CUDA_VISIBLE_DEVICES=0 nohup python run.py \
            --output_dir=savedmodels \
            --model_type=roberta \
            --config_name=../microsoft/$model \
            --model_name_or_path=../microsoft/$model \
            --tokenizer_name=../microsoft/$model \
            --do_train \
            --do_test \
            --train_data_file=../dataset/train_fuzz_clone.jsonl \
            --eval_data_file=../dataset/valid_fuzz_clone.jsonl \
            --test_data_file=../dataset/test_fuzz_clone.jsonl \
            --epoch 2 \
            --block_size 512 \
            --train_batch_size 8 \
            --eval_batch_size 16 \
            --learning_rate 2e-5 \
            --max_grad_norm 1.0 \
            --evaluate_during_training \
            --input_prompt "$input_prompt" \
            --output_prompt "$output_prompt" \
            --input_prompt_sep $input_prompt_sep \
            --need_raw $ifraw \
            --nohang \
            --nocrash \
            --n_candidate $n_candidate \
            --seed 123456 2>&1| tee ./log.txt &

wait
python ../evaluator/extract_answers.py -c ../dataset/test_fuzz_clone.jsonl -o savedmodels/answers.jsonl 
wait
python ../evaluator/evaluator.py -a savedmodels/answers.jsonl -p savedmodels/predictions.jsonl
