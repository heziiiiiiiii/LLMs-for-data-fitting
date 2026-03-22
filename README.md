# Robustness is Important: Limitations of LLMs for Data Fitting
Our Paper: ["Robustness is Important: Limitations of LLMs for Data Fitting"](https://arxiv.org/abs/2508.19563)


## References for Implementation

Below are key resources and documentation used for implementation.

### Close-weight LLMs
#### GPT-4o-mini
- OpenAI API documentation: https://platform.openai.com/docs
- Fine-tuning guide: [https://platform.openai.com/docs/guides/fine-tuning](https://platform.openai.com/docs/guides/supervised-fine-tuning)
#### Claude
- Anthropic API documentation: https://docs.anthropic.com
#### Grok
- xAI API documentation: https://docs.x.ai

### Open-weight LLMs
#### LLaMA-3-8B-Instruct
- Inference code (Hugging Face): https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct
#### Qwen
- Model page (Hugging Face): https://huggingface.co/Qwen

### Tabular Foundation Model
#### TabPFN 
- Official GitHub repository: [https://github.com/automl/TabPFN](https://github.com/PriorLabs/TabPFN)


## Code Structure

### GPT Models (`GPT/`)
- `llm.py`: Supervised fine-tuning experiments using GPT4o-mini.
- `llm_fewshot.py`: In-context learning (ICL) experiments using GPT4o-mini.

### Claude Models (`Claude/`)
- `llm_fewshot_Claude.py`: In-context learning (ICL) experiments using the Claude API.

### Grok Models (`Grok/`)
- `llm_fewshot_grok.py`: ICL experiments using the Grok API.

### LLaMA Models (`Llama/`)
- `llama3_prediction.py`: In-context learning (ICL) experiments using LLaMA-3-8B-Instruct.
- `attention_analysis_main.py`: Attention mechanism analysis for ICL10.
- `attention_analysis_fewshot20.py`: Attention mechanism analysis for ICL20.

### TabPFN (`TabPFN/`)
- `synthetic_TabPFN_linear.py`: Code to simulate linear DGP datasets for TabPFN.
- synthetic_TabPFN_original.py: Code to simulate non-linear DGP datasets for TabPFN.
- synthetic_TabPFN_logistic.py: Code to simulate logistic DGP datasets for TabPFN.
- `tabpfn.ipynb`: In-context learning (ICL) experiments using TabPFN.
- `results_summary.ipynb`: Code to aggregate results.

### Tabular Supervised Learning Techniques
- `tabular.py` : Traditional tabular supervised learning techniques (e.g., Linear Regression, Lasso, SVR, Random Forest, k-NN, MLP).

### Utilities
- `synthetic_data.py`: Code to simulate synthetic datasets.
- `utility.py` : Data loading and evaluation utilities.


