


# # do for n in 0.0, 0.2,0.4,0.6,0.8,1.0,1.2,1.4,1.6,1.8,2.0
# for n in 0.0 0.2 0.4 0.6 0.8 1.0 1.2 1.4 1.6 1.8 2.0; do
#     # create the msgpack file
#     echo "Creating msgpack file for n=${n}"
#     python analysis/analyse_logprobs.py \
#     --file_path=data/meta-llama/Llama-3.2-1B-Instruct/old_best_of_n_completions/best_of_n_completions_temp_${n}.jsonl
# done 



# for n in {3..24}; do
#     ate the msgpack file
#     echo "Creating msgpack file for n=${n}"
#     python analysis/analyse_logprobs.py \
#     --file_path=data/meta-llama/Llama-3.2-1B-Instruct/old_best_of_n_completions/best_of_n_completions-${n}.jsonl \
# }done 

# for n in {3..14}; do    
#     # analyse the msgpack file
#     echo "Analysing msgpack file for n=${n}"
#     python analysis/analyse_logprobs.py \
#     --file_path=data/meta-llama/Llama-3.2-1B-Instruct/old_best_of_n_completions/best_of_n_completions-${n}_analysis/analysis.msgpack \
#     --loop_samples \
#     --temperature=${n}  
# done







# do for n in 0.0, 0.2,0.4,0.6,0.8,1.0,1.2,1.4,1.6,1.8,2.0
# for n in 0.0 0.2 0.4 0.6 0.8 1.0 1.2 1.4 1.6 1.8 2.0; do
#     # create the msgpack file
#     echo "Creating msgpack file for n=${n}"
#     python analysis/analyse_logprobs.py \
#     --file_path=data/meta-llama/Llama-3.2-1B-Instruct/old_best_of_n_completions/best_of_n_completions_temp_${n}.jsonl
# done 

# for n in {3..14}; do
# for n in 0.0 0.2 0.4 0.6 0.8 1.0 1.2 1.4 1.6 1.8 2.0; do
#     echo "Creating msgpack file for n=${n}"
#     python analysis/analyse_logprobs.py \
#     --file_path=data/meta-llama/Llama-3.2-1B-Instruct/best_of_n_completions_temperature_ablation/best_of_n_completions_temp_${n}.jsonl
# done 

# for n in 0.0 0.2 0.4 0.6 0.8 1.0 1.2 1.4 1.6 1.8 2.0; do
#     # analyse the msgpack file
#     echo "Analysing msgpack file for n=${n}"
#     python analysis/analyse_logprobs.py \
#     --file_path=data/meta-llama/Llama-3.2-1B-Instruct/best_of_n_completions_temperature_ablation/best_of_n_completions_temp_${n}_analysis/analysis.msgpack \
#     --loop_samples \
#     --temperature=${n}  
# done



####
####
# Temperature ablation
####
####

for n_samples in 4 8 16 32 64 128 ; do
    for n in 0.0 0.2 0.4 0.6 0.8 1.0 1.2 1.4 1.6 1.8 2.0; do
        # analyse the msgpack file
        echo "Analysing msgpack file for n=${n}"
        python analysis/analyse_logprobs.py \
        --file_path=data/meta-llama/Llama-3.2-1B-Instruct/best_of_n_completions_temperature_ablation/best_of_n_completions_temp_${n}_analysis/analysis.msgpack \
        --num_tokens=16\
        --n_samples=${n_samples}\
        --temperature=${n}  
    done

    python analysis/entropy_losses_plots.py --best_of_n_completions_dir data/meta-llama/Llama-3.2-1B-Instruct/best_of_n_completions_temperature_ablation --n_samples ${n_samples}

done
