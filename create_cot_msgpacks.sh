

for i in {0..7}
do
    echo "Processing part ${i}"
    python analysis/analyse_logprobs.py data/cot_ablation/best_of_n_completions_part_${i}.jsonl 
done
