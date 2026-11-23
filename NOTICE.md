# Third-party notices and references

The code in this repository is original work by the author (see `LICENSE`). It builds on the
third-party data, research, and software listed below, which remain the property of their
respective owners. Nothing here is endorsed by any of them.

## Dataset

**RouterBench** by Martian (withmartian),
[huggingface.co/datasets/withmartian/routerbench](https://huggingface.co/datasets/withmartian/routerbench),
revision `784021482c3f320c6619ed4b3bb3b41a21424fcb`.
Companion code: [github.com/withmartian/alt-routing-methods](https://github.com/withmartian/alt-routing-methods).

- **Licence:** none is declared on the dataset card (checked 2026-11-23).
- **No redistribution:** this repository does **not** include the dataset. `make data`
  downloads it from the Hugging Face Hub, and `data/` is git-ignored. Figures and statistics in
  `docs/` are aggregate results computed from it.

> Q. J. Hu, J. Bieker, X. Li, N. Jiang, B. Keigwin, G. Ranganath, K. Keutzer,
> S. K. Upadhyay. *RouterBench: A Benchmark for Multi-LLM Routing System.* 2024.
> [arXiv:2403.12031](https://arxiv.org/abs/2403.12031)

## Benchmarks contained in RouterBench

The prompts and correctness labels come from these benchmarks; credit belongs to their authors.

| Benchmark | Reference |
|---|---|
| MMLU | Hendrycks et al., *Measuring Massive Multitask Language Understanding*, 2020. [arXiv:2009.03300](https://arxiv.org/abs/2009.03300) |
| HellaSwag | Zellers et al., *HellaSwag: Can a Machine Really Finish Your Sentence?*, 2019. [arXiv:1905.07830](https://arxiv.org/abs/1905.07830) |
| GSM8K | Cobbe et al., *Training Verifiers to Solve Math Word Problems*, 2021. [arXiv:2110.14168](https://arxiv.org/abs/2110.14168) |
| ARC | Clark et al., *Think you have Solved Question Answering? Try ARC, the AI2 Reasoning Challenge*, 2018. [arXiv:1803.05457](https://arxiv.org/abs/1803.05457) |
| WinoGrande | Sakaguchi et al., *WinoGrande: An Adversarial Winograd Schema Challenge at Scale*, 2019. [arXiv:1907.10641](https://arxiv.org/abs/1907.10641) |
| MBPP | Austin et al., *Program Synthesis with Large Language Models*, 2021. [arXiv:2108.07732](https://arxiv.org/abs/2108.07732) |
| MT-Bench | Zheng et al., *Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena*, 2023. [arXiv:2306.05685](https://arxiv.org/abs/2306.05685) |

The remaining `misc` tasks (Chinese-language tasks, `consensus_summary`, `bias_detection`,
`abstract2title`, `accounting_audit`, …) have names resembling tasks in the
[OpenAI evals](https://github.com/openai/evals) registry. Their origin is **unverified**.

## Language models

Model names (GPT-4, GPT-3.5, Claude, Llama 2, Code Llama, Mistral, Mixtral, Yi, WizardLM) appear
only as labels in the dataset. They are trademarks of their respective owners. This project does
not run, host, or redistribute any of these models.

## Software

This project uses, without modifying or redistributing them, open-source libraries including
pandas, NumPy, PyArrow, SciPy, scikit-learn, Matplotlib, MLflow, huggingface_hub, PyYAML, Jupyter,
pytest, and Ruff. Each is distributed under its own licence. See `requirements*.txt` for the exact
versions.
