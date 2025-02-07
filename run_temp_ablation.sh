
## FOR CREATING THE MSGPACK FILES
# for n in 0.0 0.2 0.4 0.6 0.8 1.0 1.2 1.4 1.6 1.8 2.0; do
#     # create the msgpack file
#     echo "Creating msgpack file for n=${n}"
#     python analysis/analyse_logprobs.py data/meta-llama/Llama-3.2-1B-Instruct/best_of_n_completions_temperature_ablation/best_of_n_completions_temp_${n}.jsonl 
# done 


## Creating the plot data

for n in 0.0 0.2 0.4 0.6 0.8 1.0 1.2 1.4 1.6 1.8 2.0; do
    python analysis/analyse_logprobs.py data/meta-llama/Llama-3.2-1B-Instruct/best_of_n_completions_temperature_ablation/best_of_n_completions_temp_${n}_analysis/analysis.msgpack 
done 