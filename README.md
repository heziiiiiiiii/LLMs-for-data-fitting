# Just Because You Can, Doesn’t Mean You Should: LLMs for Data Fitting

## References for Implementation

Below are key resources and documentation used for implementation.

### GPT-4o-mini (Closed-weight)
- OpenAI API documentation: https://platform.openai.com/docs
- Fine-tuning guide: [https://platform.openai.com/docs/guides/fine-tuning](https://platform.openai.com/docs/guides/supervised-fine-tuning)

### LLaMA-3-8B-Instruct (Open-weight)
- Inference code (Hugging Face): https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct

### TabPFN (Tabular Foundation Model)
- Official GitHub repository: [https://github.com/automl/TabPFN](https://github.com/PriorLabs/TabPFN)


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


