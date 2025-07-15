import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import torch
from datasets import Dataset, load_dataset
from loguru import logger
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)


class HuggingFaceManager:
    """
    HuggingFaceManager provides a unified interface for managing Hugging Face models, tokenisers, and datasets.

    What:
        - Loads, caches, and lists Hugging Face models and datasets.
        - Supports both online and offline workflows.
        - Handles cache directory resolution and environment configuration.
        - Offers configuration introspection and local resource discovery.
        - Supports basic text generation using the loaded model and tokeniser.

    Why:
        - Simplifies reproducible, robust LLM workflows.
        - Centralises Hugging Face infrastructure logic for maintainability and clarity.
        - Facilitates offline/online toggling and local resource management.

    How:
        - Uses environment variables and standard Hugging Face conventions.
        - Provides type hints, docstrings, and loguru logging for transparency and debugging.
        - Designed for integration in notebooks or modular codebases.

    ---------------------------------------------------------------------------
    ENVIRONMENT VARIABLES (Purpose and Defaults):

    - HF_HOME:
        * Purpose: Sets the base directory for Hugging Face cache and configuration files.
        * Default: ~/.cache/huggingface

    - TRANSFORMERS_CACHE:
        * Purpose: Sets the directory for caching Hugging Face Transformers models and tokenisers.
        * Default: $HF_HOME/transformers

    - HF_DATASETS_CACHE:
        * Purpose: Sets the directory for caching Hugging Face datasets.
        * Default: $HF_HOME/datasets

    - HUGGINGFACE_HUB_CACHE:
        * Purpose: Sets the directory for caching repositories from the Hub (models, datasets, spaces).
        * Default: $HF_HOME/hub

    - HF_HUB_OFFLINE:
        * Purpose: If set (e.g. "1"), disables all network access and forces offline mode.
        * Default: Not set (online mode)

    - HUGGING_FACE_HUB_TOKEN or HF_TOKEN:
        * Purpose: User access token for authenticating to the Hugging Face Hub.
        * Default: Not set (anonymous access)

    For further details, see: https://huggingface.co/docs/huggingface_hub/en/package_reference/environment_variables

    ---------------------------------------------------------------------------

    DEFAULT SETTINGS FOR THIS CLASS:

    - cache_dir: None (uses Hugging Face defaults)
    - offline: False (online mode by default)
    - use_gpu: False (CPU by default)
    - verbose: True (logging enabled)
    - model_name: None (no model loaded initially)
    """

    def __init__(
        self,
        cache_dir: Optional[str] = None,
        offline: bool = False,
        use_gpu: bool = False,
        verbose: bool = True,
        model_name: Optional[str] = None,
    ) -> None:
        """
        Initialises the manager and configures environment variables.

        Args:
            cache_dir: Custom cache directory. If None, uses Hugging Face defaults.
            offline: If True, enables offline mode (no network access).
            use_gpu: If True, moves models to GPU.
            verbose: If True, enables info-level logging.
            model_name: If provided, loads this model/tokeniser immediately.
        """
        self.cache_dir = cache_dir
        self.offline = offline
        self.use_gpu = use_gpu
        self.verbose = verbose
        self.model: Optional[PreTrainedModel] = None
        self.tokenizer: Optional[PreTrainedTokenizer] = None
        self.model_name: Optional[str] = None
        self._set_env_vars()
        if model_name is not None:
            self.load_model_and_tokenizer(model_name)

    def _set_env_vars(self) -> None:
        """Set environment variables for tokenizer, cache and offline mode."""
        os.environ["TOKENIZERS_PARALLELISM"] = (
            "false"  # Always disable tokenizers parallelism to avoid fork warnings
        )
        if self.cache_dir:
            os.environ["TRANSFORMERS_CACHE"] = self.cache_dir
            os.environ["HF_HOME"] = self.cache_dir
        if self.offline:
            os.environ["HF_HUB_OFFLINE"] = "1"
        else:
            os.environ.pop("HF_HUB_OFFLINE", None)
        if self.verbose:
            logger.info(f"Cache directory: {self.get_base_cache_dir()}")
            logger.info(f"Offline mode: {self.offline}")

    def get_base_cache_dir(self) -> Path:
        """
        Resolve the base Hugging Face cache directory.

        Returns:
            Path to the base cache directory.
        """
        return Path(
            self.cache_dir
            or os.environ.get("TRANSFORMERS_CACHE")
            or os.environ.get("HF_HOME")
            or Path.home() / ".cache" / "huggingface"
        )

    def get_hub_cache_dir(self) -> Path:
        """
        Return the hub subdirectory for cached models and datasets.

        Returns:
            Path to the hub cache directory.
        """
        return self.get_base_cache_dir() / "hub"

    def get_model_name(self) -> Optional[str]:
        """
        Returns the currently loaded model name, or None if not loaded.
        """
        return self.model_name

    def display_configuration_summary(self) -> Dict[str, Any]:
        """
        Print and return a summary of the Hugging Face environment configuration.

        Returns:
            Dictionary containing configuration details.
        """
        default_hf_home = str(Path.home() / ".cache" / "huggingface")
        default_transformers_cache = os.path.join(default_hf_home, "transformers")
        default_datasets_cache = os.path.join(default_hf_home, "datasets")

        transformers_cache = os.environ.get(
            "TRANSFORMERS_CACHE", default_transformers_cache
        )
        hf_home = os.environ.get("HF_HOME", default_hf_home)
        hf_datasets_cache = os.environ.get("HF_DATASETS_CACHE", default_datasets_cache)
        tokenizers_parallelism = os.environ.get("TOKENIZERS_PARALLELISM", None)

        cache_dir_display = str(self.get_base_cache_dir())

        model_id = self.model_name
        hub_url = self.get_hf_hub_url(model_id) if model_id else None

        summary = {
            "Cache Directory": cache_dir_display,
            "TRANSFORMERS_CACHE": transformers_cache,
            "HF_HOME": hf_home,
            "HF_DATASETS_CACHE": hf_datasets_cache,
            "Offline Mode": self.offline,
            "Use GPU": self.use_gpu,
            "Verbose": self.verbose,
            "Hub URL": hub_url,
            "TOKENIZERS_PARALLELISM": tokenizers_parallelism,
        }

        logger.info("Hugging Face Manager Configuration Summary:")
        for k, v in summary.items():
            logger.info(f"{k}: {v}")

        return summary

    def load_model_and_tokenizer(self, model_name: str) -> None:
        """
        Load a Hugging Face model and tokeniser, preferring cache/local files if offline.
        Stores them as instance variables.
        """
        try:
            logger.info(f"Loading model and tokeniser: {model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                cache_dir=str(self.get_base_cache_dir()),
                local_files_only=self.offline,
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                cache_dir=str(self.get_base_cache_dir()),
                local_files_only=self.offline,
            )
            if self.use_gpu and self.model is not None:
                self.model.to("cuda")
            if self.tokenizer and not self.tokenizer.pad_token:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model_name = model_name
            logger.info(f"Loaded model and tokeniser for: {model_name}")
        except Exception as e:
            logger.error(f"Failed to load model/tokeniser: {e}")
            raise

    def load_dataset(
        self, dataset_name: str, split: str = "train", **kwargs
    ) -> Dataset:
        """
        Load a Hugging Face dataset, preferring cache/local files if offline.

        Args:
            dataset_name: Name or path of the dataset.
            split: Dataset split to load (e.g. "train").
            **kwargs: Additional arguments for load_dataset.

        Returns:
            Loaded Dataset object.
        """
        try:
            logger.info(f"Loading dataset: {dataset_name} (split={split})")
            try:
                dataset = load_dataset(
                    dataset_name,
                    split=split,
                    cache_dir=str(self.get_base_cache_dir()),
                    local_files_only=self.offline,
                    **kwargs,
                )
            except (TypeError, ValueError) as e:
                if "local_files_only" in str(e):
                    logger.warning(
                        "Retrying without local_files_only (not supported by this dataset builder)."
                    )
                    dataset = load_dataset(
                        dataset_name,
                        split=split,
                        cache_dir=str(self.get_base_cache_dir()),
                        **kwargs,
                    )
                else:
                    raise
            logger.info(f"Loaded dataset: {dataset_name}")
            return dataset
        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            raise

    def check_model_downloaded(self, model_name: str) -> bool:
        """
        Check if a model is available locally in the cache.

        Args:
            model_name: Name or path of the model.

        Returns:
            True if the model is cached locally, False otherwise.
        """
        try:
            _ = AutoModelForCausalLM.from_pretrained(
                model_name,
                cache_dir=str(self.get_base_cache_dir()),
                local_files_only=True,
            )
            logger.info(f"Model '{model_name}' is available locally.")
            return True
        except Exception:
            logger.info(f"Model '{model_name}' is NOT available locally.")
            return False

    def display_dataset(self, dataset: Dataset, n: int = 3) -> None:
        """
        Display a sample of the dataset in tabular form.

        Args:
            dataset: The Hugging Face Dataset object.
            n: Number of rows to display.
        """
        rows = []
        for i in range(min(n, len(dataset))):
            example = dataset[i]
            user_msg = next(
                m["content"] for m in example["messages"] if m["role"] == "user"
            )
            assistant_msg = next(
                m["content"] for m in example["messages"] if m["role"] == "assistant"
            )
            rows.append({"User Prompt": user_msg, "Assistant Response": assistant_msg})
        df = pd.DataFrame(rows)
        pd.set_option("display.max_colwidth", None)
        logger.info(f"Displaying {n} rows from dataset.")
        display(df)

    def list_local_models(self) -> List[str]:
        """
        List all locally cached Hugging Face models.

        Returns:
            List of model names available in the local cache.
        """
        hub_dir = self.get_hub_cache_dir()
        model_dirs = sorted(hub_dir.glob("models--*"))
        models = [d.name.replace("models--", "") for d in model_dirs if d.is_dir()]
        logger.info("Locally cached models:")
        for m in models:
            logger.info(f"- {m}")
        return models

    def list_local_datasets(self) -> List[str]:
        """
        List all locally cached Hugging Face datasets.

        Returns:
            List of dataset names available in the local cache.
        """
        hub_dir = self.get_hub_cache_dir()
        dataset_dirs = sorted(hub_dir.glob("datasets--*"))
        datasets = [
            d.name.replace("datasets--", "") for d in dataset_dirs if d.is_dir()
        ]
        logger.info("Locally cached datasets:")
        for d in datasets:
            logger.info(f"- {d}")
        return datasets

    def list_local_transformers(self) -> List[str]:
        """
        Alias for listing locally cached models.

        Returns:
            List of model names available in the local cache.
        """
        return self.list_local_models()

    def generate_response(
        self,
        user_message: str,
        system_message: Optional[str] = None,
        max_new_tokens: int = 100,
    ) -> str:
        """
        Generate a response using the loaded model and tokeniser.

        Args:
            user_message: The user's message string.
            system_message: Optional system message string.
            max_new_tokens: Maximum number of new tokens to generate.

        Returns:
            The generated response as a string.
        """
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model and tokeniser must be loaded first.")

        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": user_message})

        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        input_len = inputs["input_ids"].shape[1]
        generated_ids = outputs[0][input_len:]
        response = self.tokenizer.decode(
            generated_ids, skip_special_tokens=True
        ).strip()
        logger.info("Generated response.")
        return response

    def unload_model_and_tokenizer(self) -> None:
        """
        Remove references to the loaded model and tokeniser to free memory.
        """
        self.model = None
        self.tokenizer = None
        self.model_name = None  # <-- UNSET MODEL NAME HERE
        torch.cuda.empty_cache()

    @staticmethod
    def get_hf_hub_url(model_name: str) -> Optional[str]:
        """
        Convert a Hugging Face model or dataset identifier to a Hub URL.

        Args:
            model_name: The model or dataset identifier, e.g., "Qwen/Qwen3-0.6B-Base".

        Returns:
            The corresponding Hugging Face Hub URL, or None if the format is invalid.
        """
        if not isinstance(model_name, str) or "/" not in model_name:
            return None
        return f"https://huggingface.co/{model_name}"
