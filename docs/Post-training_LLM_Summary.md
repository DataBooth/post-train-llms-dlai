# Post-training methods for LLMs Summary

## 1. Supervised Fine-Tuning (SFT)

*Teach the model to give ideal answers by showing it lots of example questions and the best possible responses.*

### Detailed Description

- **What it is:**  
  SFT is the process of further training a pre-trained LLM on a curated dataset of input-output pairs (prompts and ideal responses), usually labelled or written by humans.
- **Why it’s used:**  
  While pre-training gives the model broad language ability, SFT teaches it to follow instructions, adopt a specific style, or perform specialised tasks by mimicking high-quality examples.
- **How it works:**  
  The model is trained using standard supervised learning (cross-entropy loss), adjusting its weights to increase the likelihood of producing the target outputs when given the corresponding inputs.

### Concrete Example

- **Inputs:**  
  A dataset of pairs like:  
  - Input: “Summarise the following article: ...”  
  - Output: “This article discusses ...”
- **Method:**  
  The model is trained to map each input to its corresponding output, learning to generate the desired response when given a similar prompt.
- **Outputs:**  
  A model that, when prompted with similar instructions, produces responses that closely match the style and content of the curated examples.

## 2. Direct Preference Optimization (DPO)

*Teach the model to prefer better answers by directly learning from pairs of “good” and “bad” responses, without needing a reward model or complex reinforcement learning.*

### Detailed Description

- **What it is:**  
  DPO is a newer, simpler method for aligning LLMs with human preferences by directly optimising the model to favour responses that people prefer, based on binary preference data[1][2][3][5][6][7].
- **Why it’s used:**  
  DPO avoids the complexity and instability of traditional RLHF (which requires building a reward model and running RL). It is computationally lighter, faster, and easier to implement, yet achieves strong alignment with human values[1][2][5][6].
- **How it works:**  
  The model is trained on pairs of responses to the same prompt: one “chosen” (preferred) and one “rejected” (less preferred). The training objective directly increases the likelihood of the preferred response over the rejected one using a classification-style loss (binary cross-entropy), without building a separate reward model[1][2][3][5][6][7].

### Concrete Example

- **Inputs:**  
  For each prompt, two responses:  
  - Prompt: “Write a polite email declining an invitation.”  
    - Chosen: “Thank you for your invitation. Unfortunately, I won’t be able to attend.”  
    - Rejected: “Can’t come. Sorry.”
- **Method:**  
  The DPO algorithm adjusts the model so that it assigns higher probability to the “chosen” response than to the “rejected” one for each prompt, using only these preference pairs[1][2][3][5][6][7].
- **Outputs:**  
  A model that, when given similar prompts, is more likely to produce responses that match human preferences for tone, style, or correctness.

## 3. Online Reinforcement Learning (RL)

*Let the model try different answers, score them using a reward function (often based on human feedback), and update itself to get better scores over time.*

### Detailed Description

- **What it is:**  
  Online RL (often in the form of RLHF—Reinforcement Learning from Human Feedback) fine-tunes the LLM by having it generate outputs, scoring those outputs (with a reward model or direct human feedback), and then updating the model to maximise expected rewards.
- **Why it’s used:**  
  RL enables the model to learn complex behaviours and align with nuanced human values by optimising for long-term or subjective objectives that are hard to encode with just supervised data.
- **How it works:**  
  The process typically involves:
  1. Training a reward model to predict human preference scores for outputs.
  2. Having the LLM generate responses to prompts.
  3. Scoring those responses with the reward model.
  4. Updating the LLM’s weights using reinforcement learning algorithms (like Proximal Policy Optimization, PPO) to increase the likelihood of high-reward outputs.

### Concrete Example

- **Inputs:**  
  - Prompt: “Explain quantum computing to a 10-year-old.”  
  - Model generates several responses.
  - Human or reward model scores each response (e.g., clarity, accuracy, tone).
- **Method:**  
  The model is updated to increase the probability of generating responses that get higher scores, using RL algorithms and the reward model as a guide.
- **Outputs:**  
  A model that, over many iterations, learns to produce outputs that consistently align with the desired qualities (e.g., helpfulness, safety, clarity) as defined by the reward function.

## Summary Table

| Method      | One-Line Summary                                                                 | Inputs                                   | Method                                                                                  | Outputs                                         |
|-------------|----------------------------------------------------------------------------------|------------------------------------------|-----------------------------------------------------------------------------------------|-------------------------------------------------|
| SFT         | Mimic ideal answers from curated examples.                                       | Input-output pairs                       | Supervised learning (cross-entropy loss)                                                | Model gives responses like the examples         |
| DPO         | Prefer better answers by learning from “good” vs “bad” pairs.                    | Prompt + (chosen, rejected) response     | Directly optimise model to favour preferred responses via classification loss            | Model aligns with human preferences             |
| RL (RLHF)   | Improve by trial, scoring, and reward-driven updates.                            | Prompt, generated outputs, reward scores | Generate outputs, score with reward model/human, update using RL (e.g., PPO)             | Model optimises for high-reward, aligned output |

**References

[1] https://www.superannotate.com/blog/direct-preference-optimization-dpo
[2] https://learn.microsoft.com/en-us/azure/ai-foundry/openai/how-to/fine-tuning-direct-preference-optimization
[3] https://www.tylerromero.com/posts/2024-04-dpo/
[4] https://www.youtube.com/watch?v=k2pD3k1485A
[5] https://www.unite.ai/direct-preference-optimization-a-complete-guide/
[6] https://huggingface.co/papers/2305.18290
[7] https://arxiv.org/pdf/2305.18290.pdf
[8] https://arxiv.org/abs/2305.18290
[9] https://toloka.ai/blog/direct-preference-optimization/
[10] https://www.together.ai/blog/direct-preference-optimization