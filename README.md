# Just Because You Can, Doesn’t Mean You Should: LLMs for Data Fitting

## Code Structure

### GPT Models (`GPT/`)
- `llm.py`: Finetuning experiments with varying data representations.
- `llm_fewshot.py`: Few-shot experiments with varying data representations.

### LLaMA Models (`Llama/`)
- `llama3_prediction.py`: Performs LLaMA-3 few-shot experiments with varying data representations.
- `attention_analysis_main.py`: Attention mechanism analysis revealing position bias.
- `plot.ipynb`: Visualizations of attention patterns.

### TabPFN (`TabPFN/`)
- `tabpfn.ipynb`: Main code for running TabPFN baseline experiments.
- `synthetic_TabPFN.py`: Script for generating synthetic datasets.
- `results_summary.ipynb`: Aggregated results.


## Dataset Info

Synthetic datasets (via synthetic_data.py): linear data-generation process

## Dataset Info

- `synthetic_data.py`: Generates synthetic datasets with known functional forms (e.g., linear).
- `tabular.py`: Utilities for formatting and cleaning tabular data.
- `utility.py`: Common helper functions for prompt formatting, evaluation, etc.
