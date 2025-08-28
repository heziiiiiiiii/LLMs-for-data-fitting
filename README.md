# Just Because You Can, Doesn’t Mean You Should: LLMs for Data Fitting
[Our Paper: "Just Because You Can, Doesn’t Mean You Should: LLMs for Data Fitting"](https://arxiv.org/abs/2508.19563)


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
- `llm.py`: Supervised fine-tuning experiments using GPT4o-mini.
- `llm_fewshot.py`: In-context learning (ICL) experiments using GPT4o-mini.

### LLaMA Models (`Llama/`)
- `llama3_prediction.py`: In-context learning (ICL) experiments using LLaMA-3-8B-Instruct.
- `attention_analysis_main.py`: Attention mechanism analysis for ICL10.
- `attention_analysis_fewshot20.py`: Attention mechanism analysis for ICL20.
- `plot.ipynb`: Visualizations of attention patterns.

### TabPFN (`TabPFN/`)
- `synthetic_TabPFN.py`: Code to simulate synthetic datasets.
- `tabpfn.ipynb`: In-context learning (ICL) experiments using TabPFN.
- `results_summary.ipynb`: Code to aggregate results.

### Tabular Supervised Learning Techniques
- `tabular.py` : Traditional tabular supervised learning techniques (e.g., Linear Regression, Lasso, SVR, Random Forest, k-NN, MLP).

### Utilities
- `synthetic_data.py`: Code to simulate synthetic datasets.
- `utility.py` : Data loading and evaluation utilities.


