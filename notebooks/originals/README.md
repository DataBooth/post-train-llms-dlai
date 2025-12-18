## Original notebooks

Changes to notebooks to get them to run locally

Note that in VS Code the rendering of the information markdown cells e.g.
```
<p style="background-color:#fff6e4; padding:15px; border-width:3px; border-color:#f5ecda; border-style:solid; border-radius:6px"> ⏳ <b>Note <code>(Kernel Starting)</code>:</b> This notebook takes about 30 seconds to be ready to use. You may start and watch the video while you wait.</p>
```
is poor (presumably the choice of background colour). I removed the styling element and it renders much better in VS Code.

For each notebook I have added a description of the post-training method used.

### Lesson_3.ipynb

Removed `./models/` in front of Hugging Face model names (3 occurrences).

#### SFTConfig

For running on macOS, needed to change the `SFTConfig` to disable `bf16` and `fp16`

i.e. `bf16=False`  and `fp16=False` 

### Lesson_5.ipynb

Removed `./models/` in front of Hugging Face model names (4 occurrences).

Needed to download `helper.py` and place in the same directory as the notebook.

Removed repeated code:
```python
import warnings
warnings.filterwarnings('ignore')
```

#### `DPOConfig` turn off bf16 and fp16 for CPU only

config = DPOConfig(
    beta=0.2, 
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    num_train_epochs=1,
    learning_rate=5e-5,
    logging_steps=2,
    bf16=False,
    fp16=False,
)

### Lesson_7.ipynb

Removed `./models/` in front of Hugging Face model names (3 occurrences).


## What's good about the course?

- All the DeepLearning.AI courses are helpfully predictable following a nice "script".
- They provide the learner with a sense of what is going on without getting stuck in all the details.

## What's not so good about the course?

- The post-training tuning is implicit and somewhat mysterious as the process is performed
- 
- The local run time of some of the notebooks can we quite long without any indication of how long they will take / progress bar. This is fine for a job that runs for a minute or two, however for example in Lesson 5 this code
- [see below - just initial download of models/tokenizers that was slow - fine after re-run]:
```python
model, tokenizer = load_model_and_tokenizer("Qwen/Qwen2.5-0.5B-Instruct",
                                            USE_GPU)

test_model_with_questions(model, tokenizer, questions,
                          title="Instruct Model (Before DPO) Output")
```
took over 11 minutes to run (again seconds on re-run).

Over 26 minutes for (only seconds on re-run - possibly slow WiFi too initially?):
```python
model, tokenizer = load_model_and_tokenizer("banghua/Qwen2.5-0.5B-DPO", 
                                            USE_GPU)

test_model_with_questions(model, tokenizer, questions,
                          title="Post-trained Model (After DPO) Output")
```
Over 5 mins (although re-run after caching only a few seconds):
```python
model, tokenizer = load_model_and_tokenizer("HuggingFaceTB/SmolLM2-135M-Instruct", 
                                            USE_GPU)
```

```python
model, tokenizer = load_model_and_tokenizer("HuggingFaceTB/SmolLM2-135M-Instruct", USE_GPU)
```
38 minutes 1st run - 2nd run

## HF CLI

```bash
brew install huggingface-cli
```


