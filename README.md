# Necessary and Sufficient Watermark for LLMs


Necessary and Sufficient Watermark is a method for inserting watermarks into generated texts without degrading text quality.

## Requirements
See `setup.sh`.

## Quickstart

You can generated text by the NS-Watermark as follows:
```
prompt = <your prompt>
input_ids = tokenizer(prompt, padding=True, return_tensors="pt").input_ids

watermark = NecessaryAndSufficientWatermark(gamma=0.0001)
output_ids = watermark.generate(model, input_ids, max_length=100, alpha=1)
```

An example is provided in `examples.ipynb`.

## Experiments

You can reproduce our experiments as follows:

```
bash run.sh # C4
bash run_wmt.sh # WMT De->En
```

See `evaluation (C4).ipynb` and `evaluation (WMT).ipynb` for evaluation metrics.

For more details of commands, please refer to the help.

```
usage: evaluate_ns_watermark.py [-h] [--dataset DATASET] [--results RESULTS] [--model MODEL] [--test] [--method METHOD] [--gamma GAMMA] [--delta DELTA]
                                [--alpha ALPHA]

optional arguments:
  -h, --help         show this help message and exit
  --dataset DATASET  path of where the dataset is stored.
  --results RESULTS  path of where to save the results of the experiment.
  --model MODEL      path of where the pre-trained model is stored.
  --test             if True, the test dataset is used.
  --method METHOD    {ns_watermark, ns_watermark}.
  --gamma GAMMA      the size of green words.
  --delta DELTA      the offset used in the soft_watermark.
  --alpha ALPHA      the approximation rate used in ns_watermark.
```