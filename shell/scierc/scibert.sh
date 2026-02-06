#!/bin/bash
# GPU_ID=1
# for num in 3;do
# for seed in 45 46; do
# N=2
# M=2
# #SCIERC
# CUDA_VISIBLE_DEVICES=$GPU_ID python3 run_sure.py --model_type bertsub \
#     --model_name_or_path bert_models/scibert_scivocab_uncased --do_lower_case \
#     --data_dir scierc --num_dialogue_rounds $num \
#     --learning_rate 2e-5 --num_train_epochs 20 --per_gpu_train_batch_size 8 --per_gpu_eval_batch_size 16 --gradient_accumulation_steps 1 \
#     --max_seq_length 256 --max_pair_length 16 --save_steps 452 \
#     --seed $seed --candidate_top_n $N --candidate_worst_m $M \
#     --do_train --do_eval --evaluate_during_training --eval_all_checkpoints --eval_logsoftmax \
#     --use_typemarker \
#     --test_file /root/siton-data-guanchunxiangData/fuleihao/HGERE/saves/HGERE/scire_models/scibert/tersibcop/facencbiaf-seq512-mem400-iter3-layernorm+_attnself/ent400-rel400-lr2e-5-1e-4-bs18-ep30-eps1e-8/Hyper_scierc_scibert-$seed/pred_ent_pred_test.json \
#     --output_dir /root/siton-data-guanchunxiangData/fuleihao/SCI-NUM/out_last-N-$N-M-$M-type-num-$num-seed-$seed --overwrite_output_dir \
#     --att_left --att_right \
#     --st1_warming_up 0.2
# done;
# done;
# done;
# --output_dir scire_models/num_dialogue_rounds_$rounds-N_$N-M_$M
#--candidate_top_n $N --candidate_worst_m $M \
#--output_dir scire_models/num_dialogue_rounds_$rounds-adaptive
#--candidate_adaptive_dynamic_top --candidate_adaptive_dynamic_bottom --candidate_adaptive_include_bottom\


GPU_ID=0
for num in 3;do
for seed in 42; do
N=2
M=2
#SCIERC
CUDA_VISIBLE_DEVICES=$GPU_ID python3 run_sure.py --model_type bertsub \
    --model_name_or_path bert_models/scibert_scivocab_uncased --do_lower_case \
    --data_dir scierc --num_dialogue_rounds $num \
    --learning_rate 2e-5 --num_train_epochs 20 --per_gpu_train_batch_size 8 --per_gpu_eval_batch_size 16 --gradient_accumulation_steps 1 \
    --max_seq_length 256 --max_pair_length 16 --save_steps 452 \
    --seed $seed --candidate_top_n $N --candidate_worst_m $M \
    --do_eval --evaluate_during_training --eval_all_checkpoints --eval_logsoftmax \
    --test_file /root/siton-data-guanchunxiangData/fuleihao/HGERE/saves/HGERE/scire_models/scibert/tersibcop/facencbiaf-seq512-mem400-iter3-layernorm+_attnself/ent400-rel400-lr2e-5-1e-4-bs18-ep30-eps1e-8/Hyper_scierc_scibert-$seed/pred_ent_pred_test.json \
    --output_dir /root/siton-data-guanchunxiangData/fuleihao/SCI-NUM/N-$N-M-$M-type-num-$num-seed-$seed --overwrite_output_dir \
    --att_left --att_right --use_ner_results\
    --st1_warming_up 0.2
done;
done;