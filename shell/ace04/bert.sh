GPU_ID=1
N=2
M=2
for num in 1;do
for seed in 44 45 46; do 
for flod in 0 1 2 3 4; do
CUDA_VISIBLE_DEVICES=$GPU_ID  python3  run_sure.py  --model_type bertsub  \
    --model_name_or_path  bert_models/bert-base-uncased  --do_lower_case  \
    --data_dir ace04  \
    --learning_rate 2e-5  --num_train_epochs 30  --per_gpu_train_batch_size  8  --per_gpu_eval_batch_size 16  --gradient_accumulation_steps 1  \
    --max_seq_length 256  --max_pair_length 32  --save_steps 999999  \
    --do_train --do_eval  --evaluate_during_training   --eval_all_checkpoints  --eval_logsoftmax  \
    --seed $seed --use_typemarker --num_dialogue_rounds $num \
    --candidate_top_n $N --candidate_worst_m $M\
    --train_file train/$flod.json  --dev_file train/$flod.json \
    --test_file /root/siton-data-guanchunxiangData/fuleihao/HGERE/saves/HGERE/ace04re_models/bert/seed-$seed/Hyper_ace04_bert-fold$flod/pred_ent_pred_test_$flod.json  \
    --output_dir ACE04-RE/seed-$seed/ace04re-bert-$flod  --overwrite_output_dir \
    --att_left --att_right \
    --st1_warming_up 0.2
done;
done;
done;


# GPU_ID=1
# N=2
# M=2
# for num in 3;do
# for seed in 42; do 
# for flod in 0 ; do
# CUDA_VISIBLE_DEVICES=$GPU_ID  python3  run_sure.py  --model_type bertsub  \
#     --model_name_or_path  bert_models/bert-base-uncased  --do_lower_case  \
#     --data_dir ace04  \
#     --learning_rate 2e-5  --num_train_epochs 30  --per_gpu_train_batch_size  8  --per_gpu_eval_batch_size 16  --gradient_accumulation_steps 1  \
#     --max_seq_length 256  --max_pair_length 32  --save_steps 999999  \
#     --do_eval  --evaluate_during_training   --eval_all_checkpoints  --eval_logsoftmax  \
#     --seed $seed --use_typemarker --num_dialogue_rounds $num \
#     --candidate_top_n $N --candidate_worst_m $M\
#     --train_file train/$flod.json  --dev_file train/$flod.json \
#     --test_file /root/siton-data-guanchunxiangData/fuleihao/HGERE/saves/HGERE/ace04re_models/bert/seed-$seed/Hyper_ace04_bert-fold$flod/pred_ent_pred_test_$flod.json  \
#     --output_dir ACE04-RE/seed-$seed/ace04re-bert-$flod  --overwrite_output_dir \
#     --att_left --att_right \
#     --st1_warming_up 0.2
# done;
# done;
# done;