# Just Because You Can, Doesn’t Mean You Should: LLMs for Data Fitting

## Models
We use three types models:

close-weight general-purpose LLMs: GPT40-mini

p[en-weight general-purpose LLMs: Llama-3-8B-instruct

special purpose tabular foundation model: TabPFN

## GPT Models
llm.py: Core GPT implementation adapted for tabular data prediction

llm_fewshot.py: In-context learning experiments with various data representations

## Llama Models

llama3_prediction.py: Llama3 prediction on tabular data

attention_analysis_main.py: Detailed attention mechanism analysis revealing position bias

plot.ipynb: Visualization of attention patterns 

## TabPFN (Tabular Foundation Model)

tabpfn.ipynb: Baseline TabPFN performance evaluation

synthetic_TabPFN.py: generate

results_summary.ipynb: Cross-model robustness comparison
