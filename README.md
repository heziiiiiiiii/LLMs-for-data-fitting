# Just Because You Can, Doesn’t Mean You Should: LLMs for Data Fitting

## Code Structure

### GPT Models (`GPT/`)
- `llm.py`: Fine-tuning experiments using GPT4o-mini on synthetic datasets.
- `llm_fewshot.py`: Few-shot experiments (ICL) using GPT4o-mini on synthetic datasets.

### LLaMA Models (`Llama/`)
- `llama3_prediction.py`: Few-shot experiments (ICL) using LLaMA-3-8B-Instruct on synthetic datasets.
- `attention_analysis_main.py`: Attention mechanism analysis.
- `plot.ipynb`: Visualizations of attention patterns.

### TabPFN (`TabPFN/`)
- `synthetic_TabPFN.py`: Code for creating synthetic datasets.
- `tabpfn.ipynb`: Few-shot experiments (ICL) using TabPFN on synthetic datasets.
- `results_summary.ipynb`: Code for aggregating results.

### Tabular Supervised Learning Techniques
- `tabular.py` : Traditional tabular supervised learning techniques (e.g., Linear Regression, Lasso, SVR, Random Forest, k-NN, MLP) on synthetic datasets.

### Utilities
- `synthetic_data.py`: Code for creating synthetic datasets.
- `utility.py` : Data loading and evaluation utilities.


