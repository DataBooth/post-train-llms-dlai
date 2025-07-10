# post-train-llms-dlai

Post training of LLMs - Deep Learning AI course

This repository contains Jupyter notebooks and resources accompanying my completion of the [DeepLearning.AI Post-training of LLMs](https://www.deeplearning.ai/short-courses/post-training-of-llms/) course. These materials explore hands-on techniques for post-training large language models (LLMs), including Supervised Fine-Tuning (SFT), Direct Preference Optimization (DPO), and Online Reinforcement Learning (RL).

<!-- TODO: Marimo versions? Some refactoring of the notebooks for clarity e.g. docstrings, comments, and structure. -->

## About

The notebooks here are designed to help you understand and experiment with the core methods that take a pre-trained LLM from “generalist” to “specialist,” making it more useful, reliable, and aligned with human intent.

- **Supervised Fine-Tuning (SFT):** Training models on curated prompt-response pairs.
- **Direct Preference Optimization (DPO):** Aligning models using preferred vs. rejected outputs.
- **Online RL:** Iteratively improving model outputs using reward signals.

The examples use accessible models (like `HuggingFaceTB/SmolLM2-135M`) and small datasets so you can run the full training process even on modest hardware. If you have a GPU, you can experiment with larger models such as `Qwen/Qwen3-0.6B-Base` for more advanced results[1].

## Notebooks

- `Lesson_3.ipynb`: Introduction to SFT and dataset preparation
- `Lesson_4.ipynb`: Implementing DPO for model alignment
- `Lesson_5.ipynb`: Full SFT workflow with a small model and dataset ([see sample](Lesson_3.ipynb))[1]
- `Lesson_6.ipynb`: Online RL basics and reward modeling

Each notebook is self-contained and includes comments to guide you through the process.

## Getting Started

1. **Clone the repo**  
   ```
   git clone https://github.com/your-username/llm-post-training-course-notebooks.git
   cd llm-post-training-course-notebooks
   ```

2. **Set up your environment**  
   - Recommended: Python 3.12+, Jupyter, and [Hugging Face Transformers](https://huggingface.co/docs/transformers/index).
   - Install dependencies (see `pyproject.toml` for details):
     ```
     uv venv && uv sync
     ```

3. **Run the notebooks**  
   - Launch Jupyter Notebook and open any notebook to start exploring.

## Why Post-training?

Post-training is what makes LLMs genuinely useful for real-world applications—aligning them with human preferences, improving safety, and customizing for specific business needs. This repo is part of my ongoing commitment to staying at the forefront of AI advancements and bringing practical, cutting-edge solutions to client projects. For more on why this matters, see my write-up: [Post-training of LLMs: What, Why, and How](https://www.databooth.com.au/posts/post-training-llms/).

## Certificate

I’ve completed the DeepLearning.AI course on Post-training of LLMs—see my certificate [here](https://learn.deeplearning.ai/accomplishments/your-certificate-link).

## License

MIT License. See [LICENSE](LICENSE) for details.

---

*If you have feedback or want to discuss practical applications of LLM post-training, feel free to reach out!*