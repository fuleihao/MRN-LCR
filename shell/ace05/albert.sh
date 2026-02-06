GPU_ID=1
for num in 3;do
N=2
M=2
for seed in 43; do 
CUDA_VISIBLE_DEVICES=$GPU_ID  python3  run_sure.py  --model_type albertsub  \
    --model_name_or_path  bert_models/albert-xxlarge-v1  --do_lower_case  \
    --data_dir ace05 --num_dialogue_rounds $num \
    --learning_rate 2e-5  --num_train_epochs 10  --per_gpu_train_batch_size  8  --per_gpu_eval_batch_size 16  --gradient_accumulation_steps 1  \
    --max_seq_length 256  --max_pair_length 32  --save_steps 3362  \
    --do_train --do_eval  --evaluate_during_training   --eval_all_checkpoints  --eval_logsoftmax  \
    --seed $seed --use_typemarker \
    --candidate_top_n $N --candidate_worst_m $M \
    --test_file /root/siton-data-guanchunxiangData/fuleihao/HGERE/saves/HGERE/ace05re_models/bert/tersibcop/facencbiaf-seq256-mem400-iter3-eps1e-8-layernorm+-attnself/ent400-rel400-lr2e-5-5e-5-bs18-ep15/Hyper_ace05_bert-$seed/ent_pred_test.json  \
    --output_dir ACE05-RE/ace05re-albert-$seed  --overwrite_output_dir \
    --att_left --att_right \
    --st1_warming_up 0.2
done;
done;




# GPU_ID=0
# for num in 3;do
# N=2
# M=2
# for seed in 42; do 
# CUDA_VISIBLE_DEVICES=$GPU_ID  python3  run_sure.py  --model_type albertsub  \
#     --model_name_or_path  bert_models/albert-xxlarge-v1  --do_lower_case  \
#     --data_dir ace05  \
#     --learning_rate 2e-5  --num_train_epochs 10  --per_gpu_train_batch_size  8  --per_gpu_eval_batch_size 16  --gradient_accumulation_steps 1  \
#     --max_seq_length 256  --max_pair_length 32  --save_steps 3362  \
#     --do_eval  --evaluate_during_training   --eval_all_checkpoints  --eval_logsoftmax  \
#     --seed $seed --use_typemarker --use_ner_result\
#     --candidate_top_n $N --candidate_worst_m $M \
#     --test_file /root/siton-data-guanchunxiangData/fuleihao/HGERE/saves/HGERE/ace05re_models/bert/tersibcop/facencbiaf-seq256-mem400-iter3-eps1e-8-layernorm+-attnself/ent400-rel400-lr2e-5-5e-5-bs18-ep15/Hyper_ace05_bert-$seed/ent_pred_test.json  \
#     --output_dir ACE05-RE/ace05re-albert-$seed  --overwrite_output_dir \
#     --att_left --att_right \
#     --st1_warming_up 0.2
# done;
# done;


