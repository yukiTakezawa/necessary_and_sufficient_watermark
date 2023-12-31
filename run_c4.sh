export CUDA_VISIBLE_DEVICES="0"

model="/data1/takezawa/huggingface/"
dataset="/data1/takezawa/c4/realnewslike"

alpha=1


# Generates texts by the NS-Watermark.
for gamma in {0.0001,} ; do
    mkdir -p results/c4
    python evaluate_ns_watermark.py --method ns_watermark --gamma ${gamma} --alpha ${alpha} --results results/c4/ns_watermark_${gamma}_${alpha}.csv.gz --model ${model} --dataset ${dataset}
done		 

# compute z-scores of texts generated by the NS-Watermark.
for gamma in {0.0001,} ; do
    mkdir -p results/c4/z_score
    python compute_z_score_llama.py --in_results results/c4/ns_watermark_${gamma}_${alpha}.csv.gz --out_results  results/c4/z_score/ns_watermark_${gamma}_${alpha}.csv.gz --gamma ${gamma} --model ${model}
done

# compute z-scores of texts written by humans.
for gamma in {0.0001,} ; do
    mkdir -p results/c4/z_score
    python compute_z_score_llama.py --in_results results/c4/ns_watermark_${gamma}_${alpha}.csv.gz --out_results  results/c4/z_score/human_${gamma}.csv.gz --gamma ${gamma} --model ${model} --human
done


# compute PPL.
for gamma in {0.0001,} ; do
    mkdir -p results/c4/z_score
    python compute_ppl.py --results results/c4/ns_watermark_${gamma}_${alpha}.csv.gz --gamma ${gamma} --model ${model}
done


"""
delta=6
# Generates texts by the NS-Watermark.
for gamma in {0.1,} ; do
    mkdir -p results/c4
    python evaluate_ns_watermark.py --method soft_watermark --gamma ${gamma} --alpha ${alpha} --results results/c4/soft_watermark_${gamma}_${delta}.csv.gz --model ${model} --dataset ${dataset} --delta ${delta}
done		 

# compute z-scores of texts generated by the NS-Watermark.
for gamma in {0.1,} ; do
    mkdir -p results/c4/z_score
    python compute_z_score_llama.py --in_results results/c4/soft_watermark_${gamma}_${delta}.csv.gz --out_results  results/c4/z_score/soft_watermark_${gamma}_${delta}.csv.gz --gamma ${gamma} --model ${model} 
done
"""
