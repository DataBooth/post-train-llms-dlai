import marimo

__generated_with = "0.14.10"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    import torch
    from typing import Optional, List, Tuple
    from loguru import logger
    import pandas as pd
    from transformers import PreTrainedTokenizer, PreTrainedModel, AutoTokenizer, AutoModelForCausalLM
    from datasets import load_dataset
    from trl import SFTTrainer, SFTConfig
    return (
        AutoModelForCausalLM,
        AutoTokenizer,
        List,
        Optional,
        PreTrainedModel,
        PreTrainedTokenizer,
        SFTConfig,
        SFTTrainer,
        Tuple,
        load_dataset,
        logger,
        pd,
        torch,
    )


@app.cell
def _(Optional, PreTrainedModel, PreTrainedTokenizer, logger, torch):
    def generate_responses(
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        user_message: str,
        system_message: Optional[str] = None,
        max_new_tokens: int = 100
    ) -> str:
        """
        Generate a model response for a user (and optional system) message.

        Args:
            model: The language model to use for generation.
            tokenizer: Tokenizer corresponding to the model.
            user_message: The user's input message.
            system_message: Optional system prompt for context.
            max_new_tokens: Maximum number of tokens to generate.

        Returns:
            The assistant's generated response as a string.
        """
        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": user_message})

        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

        try:
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            input_len = inputs["input_ids"].shape[1]
            generated_ids = outputs[0][input_len:]
            response = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
            logger.debug(f"Generated response: {response}")
            return response
        except Exception as e:
            logger.error(f"Error during response generation: {e}")
            return ""
    return (generate_responses,)


@app.cell
def _(
    List,
    Optional,
    PreTrainedModel,
    PreTrainedTokenizer,
    generate_responses,
    logger,
):
    def try_model_with_questions(
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        questions: List[str],
        system_message: Optional[str] = None,
        title: str = "Model Output"
    ) -> None:
        """
        Print model responses for a list of questions.

        Args:
            model: The language model for generation.
            tokenizer: Corresponding tokenizer.
            questions: List of user questions.
            system_message: Optional system prompt.
            title: Section title for output.
        """
        logger.info(f"=== {title} ===")
        for i, question in enumerate(questions, 1):
            response = generate_responses(model, tokenizer, question, system_message)
            logger.info(f"\nModel Input {i}:\n{question}\nModel Output {i}:\n{response}\n")
    return (try_model_with_questions,)


@app.cell
def _(
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    Tuple,
    logger,
):
    def load_model_and_tokenizer(
        model_name: str,
        use_gpu: bool = False
    ) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
        """
        Load a model and tokenizer from a local path or Hugging Face Hub.

        Args:
            model_name: Path or identifier of the model.
            use_gpu: Whether to move model to GPU.

        Returns:
            Tuple of (model, tokenizer).
        """
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(model_name)
            logger.info(f"Loaded model and tokenizer from {model_name}")
        except Exception as e:
            logger.error(f"Error loading model or tokenizer: {e}")
            raise

        if use_gpu:
            model.to("cuda")

        if not tokenizer.chat_template:
            tokenizer.chat_template = """{% for message in messages %}
                {% if message['role'] == 'system' %}System: {{ message['content'] }}\n
                {% elif message['role'] == 'user' %}User: {{ message['content'] }}\n
                {% elif message['role'] == 'assistant' %}Assistant: {{ message['content'] }} <|endoftext|>
                {% endif %}
                {% endfor %}"""

        if not tokenizer.pad_token:
            tokenizer.pad_token = tokenizer.eos_token

        return model, tokenizer
    return (load_model_and_tokenizer,)


@app.cell
def _(display, logger, pd):
    def display_dataset(dataset) -> None:
        """
        Display the first three user/assistant pairs in the dataset.

        Args:
            dataset: Hugging Face Dataset object with 'messages'.
        """
        rows = []
        for i in range(3):
            example = dataset[i]
            user_msg = next(m['content'] for m in example['messages'] if m['role'] == 'user')
            assistant_msg = next(m['content'] for m in example['messages'] if m['role'] == 'assistant')
            rows.append({
                'User Prompt': user_msg,
                'Assistant Response': assistant_msg
            })
        df = pd.DataFrame(rows)
        pd.set_option('display.max_colwidth', None)
        display(df)
        logger.info("Displayed first 3 examples from dataset")

    return (display_dataset,)


@app.cell
def _(mo):
    mo.md(
        r"""
    ## SFT Workflow: Model Loading, Testing, and Training

    This section demonstrates loading base and SFT models, evaluating them, and running SFT on a small model and dataset.
    """
    )
    return


@app.cell
def _(List, logger):
    USE_GPU: bool = False  # Set to True if you want to use GPU

    questions: List[str] = [
        "Give me an 1-sentence introduction of LLM.",
        "Calculate 1+1-1",
        "What's the difference between thread and process?"
    ]

    logger.info(f"Device set to {'GPU' if USE_GPU else 'CPU'}")
    logger.info(f"Test questions: {questions}")
    return USE_GPU, questions


@app.cell
def _(mo):
    mo.md(r"""## 2. Load and Test Base Model (Before SFT)""")
    return


@app.cell
def _(
    USE_GPU: bool,
    load_model_and_tokenizer,
    logger,
    questions: "List[str]",
    try_model_with_questions,
):
    try:
        base_model, base_tokenizer = load_model_and_tokenizer("Qwen/Qwen3-0.6B-Base", USE_GPU)
        logger.info("Loaded base model for pre-SFT evaluation.")
        try_model_with_questions(base_model, base_tokenizer, questions, title="Base Model (Before SFT) Output")
    except Exception as e:
        logger.error(f"Failed to load/test base model: {e}")
    finally:
        del base_model, base_tokenizer
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## 3. Load and Test SFT Model (After SFT)

    This section demonstrates the results of a previously fine-tuned (SFT) model.
    """
    )
    return


@app.cell
def _(
    USE_GPU: bool,
    load_model_and_tokenizer,
    logger,
    questions: "List[str]",
    try_model_with_questions,
):
    def load_test_after_sft(questions):
        try:
            model, tokenizer = load_model_and_tokenizer("./models/banghua/Qwen3-0.6B-SFT", USE_GPU)
            logger.info("Loaded SFT model for post-SFT evaluation.")
            try_model_with_questions(model, tokenizer, questions, title="Base Model (After SFT) Output")
        except Exception as e:
            return logger.error(f"Failed to load/test SFT model: {e}")
        finally:
            del model, tokenizer
            return None

    load_test_after_sft(questions)
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## 4. SFT on a Small Model and Dataset

    We use a small model (`HuggingFaceTB/SmolLM2-135M`) and a subset of a dataset for demonstration and resource efficiency.
    """
    )
    return


@app.cell
def _(USE_GPU: bool, load_model_and_tokenizer, logger):
    model_name: str = "./models/HuggingFaceTB/SmolLM2-135M"
    try:
        model, tokenizer = load_model_and_tokenizer(model_name, USE_GPU)
        logger.info(f"Loaded small model '{model_name}' for SFT.")
    except Exception as e:
        logger.error(f"Failed to load small model for SFT: {e}")
        raise
    return model, tokenizer


@app.cell
def _(mo):
    mo.md(r"""Load and preview training dataset""")
    return


@app.cell
def _(USE_GPU: bool, display_dataset, load_dataset, logger):
    try:
        train_dataset = load_dataset("banghua/DL-SFT-Dataset")["train"]
        if not USE_GPU:
            train_dataset = train_dataset.select(range(100))  # Use a subset for CPU training
        logger.info(f"Loaded training dataset with {len(train_dataset)} samples.")
        display_dataset(train_dataset)
    except Exception as e:
        logger.error(f"Failed to load or display training dataset: {e}")
        raise
    return (train_dataset,)


@app.cell
def _(mo):
    mo.md(r"""### SFT Trainer Configuration""")
    return


@app.cell
def _(SFTConfig, logger):
    # SFT training configuration
    sft_config = SFTConfig(
        learning_rate=8e-5,
        num_train_epochs=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        gradient_checkpointing=False,
        logging_steps=2,
    )
    logger.info("SFTConfig initialized.")
    return (sft_config,)


@app.cell
def _(mo):
    mo.md(r"""### Run SFT Training""")
    return


@app.cell
def _(SFTTrainer, logger, model, sft_config, tokenizer, train_dataset):
    try:
        sft_trainer = SFTTrainer(
            model=model,
            args=sft_config,
            train_dataset=train_dataset,
            processing_class=tokenizer,
        )
        logger.info("SFTTrainer initialized. Starting training...")
        sft_trainer.train()
        logger.success("SFT training completed successfully.")
    except Exception as e:
        logger.error(f"Error during SFT training: {e}")
        raise
    return (sft_trainer,)


@app.cell
def _(mo):
    mo.md(r"""## 5. Evaluate Fine-Tuned Model on Example Questions""")
    return


@app.cell
def _(
    USE_GPU: bool,
    logger,
    questions: "List[str]",
    sft_trainer,
    tokenizer,
    try_model_with_questions,
):
    try:
        if not USE_GPU:
            sft_trainer.model.to("cpu")
        logger.info("Evaluating fine-tuned model on test questions...")
        try_model_with_questions(sft_trainer.model, tokenizer, questions, title="Base Model (After SFT) Output")
    except Exception as e:
        logger.error(f"Error during evaluation of fine-tuned model: {e}")

    return


if __name__ == "__main__":
    app.run()
