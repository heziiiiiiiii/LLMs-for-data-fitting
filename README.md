# Robustness is Important: Limitations of LLMs for Data Fitting
Our Paper: ["Robustness is Important: Limitations of LLMs for Data Fitting"](https://arxiv.org/abs/2508.19563)


## References for Implementation

Below are key resources and documentation used for implementation.

### Close-weight LLMs
#### GPT-4o-mini
- OpenAI API documentation: https://platform.openai.com/docs
- Fine-tuning guide: [https://platform.openai.com/docs/guides/fine-tuning](https://platform.openai.com/docs/guides/supervised-fine-tuning)
#### Claude-4.5-Sonnet
- Anthropic API documentation: https://docs.anthropic.com
#### Grok-4.1
- xAI API documentation: https://docs.x.ai

### Open-weight LLMs
#### LLaMA-3-8B-Instruct
- Inference code (Hugging Face): https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct
#### Qwen-3-4B
- Model page (Hugging Face): https://huggingface.co/Qwen

### Tabular Foundation Models
#### TabPFN 
- Official GitHub repository: [https://github.com/automl/TabPFN](https://github.com/PriorLabs/TabPFN)
#### LimiX
- Official ducumentation: https://www.limix.ai/doc


## Code Structure

### GPT Models (`GPT/`)
- `llm.py`: Supervised fine-tuning experiments using GPT4o-mini.
- `llm_fewshot.py`: In-context learning (ICL) experiments using GPT4o-mini.

### Claude Models (`Claude/`)
- `llm_fewshot_Claude.py`: In-context learning (ICL) experiments using Claude-4.5-Sonnet (Anthropic).

### Grok Models (`Grok/`)
- `llm_fewshot_grok.py`: In-context learning (ICL) experiments using Grok-4.1 (xAI).

### LLaMA Models (`Llama/`)
- `llama3_prediction.py`: In-context learning (ICL) experiments using LLaMA-3-8B-Instruct.
- `attention_analysis_main.py`: Attention mechanism analysis for ICL10.
- `attention_analysis_fewshot20.py`: Attention mechanism analysis for ICL20.
- `llama3_prediction_outlier.py`: Outliers analysis.

### Qwen Models (`Qwen/`)
- `qwen3_prediction.py`: In-context learning (ICL) experiments using Qwen-3-4B.
- `attention_analysis_qwen.py`: Attention mechanism analysis for Qwen ICL10.

### TabPFN (`TabPFN/`)
- `synthetic_TabPFN_linear.py`: Code to simulate linear DGP datasets for TabPFN.
- `synthetic_TabPFN_nonlinear.py`: Code to simulate nonlinear DGP datasets for TabPFN.
- `synthetic_TabPFN_logistic.py`: Code to simulate logistic DGP datasets for TabPFN.
- `tabpfn_v2.py`: In-context learning (ICL) experiments for linear and nonlinear DGPs using TabPFN v2.
- `tabpfn_v2_5.py`: In-context learning (ICL) experiments for linear and nonlinear DGPs using TabPFN v2.5.
- `tabpfn_logistic_v2.py`: In-context learning (ICL) experiments for logistic DGP using TabPFN v2.
- `tabpfn_logistic_v2_5.py`: In-context learning (ICL) experiments for logistic DGP using TabPFN v2.5.
- `results_summary.ipynb`: Code to aggregate results.

### LimiX (`LimiX/`)
- `classification.py`: Binary classification experiments using LimiX on the logistic DGP.
- `regression.py`: Numeric prediction experiments using LimiX on linear and nonlinear DGPs.

### Tabular Supervised Learning Techniques
- `tabular.py` : Traditional tabular supervised learning techniques (e.g., Linear Regression, Lasso, SVR, Random Forest, k-NN, MLP).

### Utilities
- `synthetic_linear.py`: Generates synthetic datasets with a linear DGP.
- `synthetic_logistic.py`: Generates synthetic datasets with a logistic DGP.
- `synthetic_nonlinear.py`: Generates synthetic datasets with a nonlinear DGP.
- `utility.py` : Data loading and evaluation utilities.


