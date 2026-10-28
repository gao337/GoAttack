# GoAttack

Anonymous code artifact for the AAAI 2027 submission **"GoAttack: A Genetic-optimization Attack on Large Reasoning Models via Adaptive Reasoning Chain Injection."**

GoAttack is a black-box attack on large reasoning models. It creates alternative questions with different answers, extracts misleading reasoning prefixes, and uses genetic optimization to jointly improve attack efficacy, reasoning coherence, and semantic stealthiness. The artifact includes the four datasets used in the paper: MATH, GSM8K, CSQA, and MATHQA.

> This code is provided only for reproducible research on model robustness. Use it only with models and endpoints that you are authorized to evaluate.

## 1. Environment

The released environment uses Python 3.10.19. The main dependencies are
pinned in `requirements.txt` using versions from the `document` environment;
their transitive dependencies are installed automatically by `pip`.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

On Windows PowerShell, activate the environment with:

```powershell
.venv\Scripts\Activate.ps1
```

### Sentence encoder

To keep the anonymous artifact below the 50 MB submission limit, the
approximately 90 MB `all-MiniLM-L6-v2` weights may be omitted from the ZIP.
When the local model directory is absent, GoAttack automatically downloads
`sentence-transformers/all-MiniLM-L6-v2` from Hugging Face on first use and
caches it in:

```text
models--sentence-transformers--all-MiniLM-L6-v2/
```

Internet access is needed only for this first download. Reviewers may download
the model before running an experiment:

```bash
python -c "from model_utils import load_embedding_model; load_embedding_model()"
```

For an offline machine, run the command above on a connected machine and copy
the resulting model directory into the repository root. If `HF_HUB_OFFLINE=1`
is set, disable it during the download. A custom local directory can be selected
with `GOATTACK_EMBEDDING_MODEL_DIR`.

The code calls an OpenAI-compatible Chat Completions endpoint. The endpoint must provide both the assistant model and the victim model used in a run.

```bash
export OPENAI_API_KEY="your-api-key"
export OPENAI_BASE_URL="https://your-endpoint.example/v1"
```

PowerShell equivalent:

```powershell
$env:OPENAI_API_KEY="your-api-key"
$env:OPENAI_BASE_URL="https://your-endpoint.example/v1"
```

Use the model identifiers exposed by your endpoint. The paper uses Qwen2.5-7B-Instruct as the assistant model and evaluates Qwen3-30B-A3B, O3-mini, DeepSeek-V4-Flash, and Gemini-3.1-Flash-Lite as victim models. API calls may incur provider charges.

## 2. Quick reviewer run

Run all commands from the repository root. The following example evaluates the zero-shot MATH setting with the paper's main hyperparameters. Replace the model identifiers if your endpoint uses different names.

```bash
# 1. Construct alternative questions.
python QuestionModification.py \
  --dataset MATH \
  --llm_name qwen2.5-7b-instruct \
  --seed 0

# 2. Generate reasoning chains for the alternative questions.
python GetSolutionofQuestionModified.py \
  --dataset MATH \
  --llm_name qwen2.5-7b-instruct \
  --seed 0

# 3. Obtain clean, no-attack predictions.
python NoAttack.py \
  --dataset MATH \
  --modify_llm_name qwen2.5-7b-instruct \
  --llm_name qwen3-30b-a3b \
  --seed 0

# 4. Run GoAttack.
python GoAttack.py \
  --dataset MATH \
  --modify_llm_name qwen2.5-7b-instruct \
  --llm_name qwen3-30b-a3b \
  --ratio 0.6 \
  --population_size 6 \
  --max_generations 3 \
  --seed 0
```

`QuestionModification.py` is capped at 10 examples to keep the reviewer run short. The paper reports experiments on 300 randomly sampled examples from each dataset. For a full-scale rerun, use the same released data and seed, change the cap in `QuestionModification.py` from 10 to 300, and then execute the same pipeline.

Valid dataset names are `MATH`, `gsm8k`, `csqa`, and `MATHQA`. To run the few-shot setting, append `--few_shot True` to steps 2-4. Omit this flag for zero-shot evaluation.

### Coherence-loss variant

`GoAttackCoherence.py` reuses the existing `GoAttack.py` pipeline but replaces
the keyword-based rebuttal loss with an LLM-judged local coherence loss
`L_coherence = 1 - (C - 1) / 4`, where `C ∈ {1..5}` is scored by the assistant
model on the transition from the injected reasoning prefix to the model's first
continuation step. The fitness uses `w1=1, w2=1, w3=0.5` with no success
bonus. Its early-stop threshold is -0.35, equivalent to "attack succeeds,
C >= 4, semantic similarity >= 0.8".

```bash
python GoAttackCoherence.py \
  --dataset MATH \
  --modify_llm_name qwen2.5-7b-instruct \
  --llm_name qwen3-30b-a3b \
  --ratio 0.6 \
  --population_size 6 \
  --max_generations 3 \
  --seed 0
```

It reads the same intermediate files as `GoAttack.py` and writes to
`final_solution_modified_q_CoT_ga_coherence/` (or the `_few_shot_` variant).

## 3. Evaluation

Compute clean accuracy, attacked accuracy, attack success rate, token statistics, and semantic similarity:

```bash
python Evaluation.py \
  --dataset MATH \
  --modify_llm_name qwen2.5-7b-instruct \
  --llm_name qwen3-30b-a3b \
  --ratio 0.6 \
  --population_size 6 \
  --max_generations 3 \
  --ga True \
  --num .
```

The paper's Coherence Score is obtained with an LLM judge (GPT-4o in the reported experiments):

```bash
python CoherenceEvaluator.py \
  --dataset MATH \
  --modify_llm_name qwen2.5-7b-instruct \
  --llm_name qwen3-30b-a3b \
  --ratio 0.6 \
  --population_size 6 \
  --max_generations 3 \
  --ga True \
  --num . \
  --judge_llm gpt-4o \
  --judge_api_key "$OPENAI_API_KEY"
```

For the coherence-loss variant, pass `--ablation coherence` to both evaluators
so they read the `final_solution_modified_q_CoT_ga_coherence` directory:

```bash
python Evaluation.py \
  --dataset MATH \
  --modify_llm_name qwen2.5-7b-instruct \
  --llm_name qwen3-30b-a3b \
  --ratio 0.6 \
  --population_size 6 \
  --max_generations 3 \
  --ga True \
  --ablation coherence \
  --num .

python CoherenceEvaluator.py \
  --dataset MATH \
  --modify_llm_name qwen2.5-7b-instruct \
  --llm_name qwen3-30b-a3b \
  --ratio 0.6 \
  --population_size 6 \
  --max_generations 3 \
  --ga True \
  --ablation coherence \
  --num . \
  --judge_llm gpt-4o \
  --judge_api_key "$OPENAI_API_KEY"
```

Main output files are written under:

```text
modified_question/<dataset>/<assistant-model>/
raw_solution_modified_q_CoT/<dataset>/<assistant-model>/
final_solution_modified_q_CoT/<dataset>/<assistant-model>/<victim-model>/
final_solution_modified_q_CoT_ga/<dataset>/<assistant-model>/<victim-model>/
final_solution_modified_q_CoT_ga_coherence/<dataset>/<assistant-model>/<victim-model>/
```

Existing JSONL output is detected by the main attack and baseline scripts, allowing interrupted runs to resume.

## 4. Experimental configuration

- Population size: 6
- Maximum generations: 3
- Elite size: 2
- Fitness weights used by the released code (w1 coherence, w2 attack efficacy, w3 stealthiness): (1, 1, 0.5)
- Early-stop thresholds used by the released code: -0.1 for the main GA; -0.35 (= -(w1*0.25 + w3*0.2)) for the coherence GA
- Reasoning-prefix injection ratio: 0.6
- Evaluation size: 300 examples per dataset
- Prompting: zero-shot and three-example few-shot
- Metrics: Accuracy (ACC), Attack Success Rate (ASR), Coherence Score (CS), and Semantic Similarity (SS)
- Timing hardware reported in the paper: one NVIDIA GeForce RTX 4090 with 24 GB VRAM

The 50 MB review package excludes the sentence-encoder weights. They are
downloaded and cached automatically as described in Section 1. The downloaded
model is required for the paper's semantic-stealthiness objective and SS metric.

## 5. Repository layout

```text
GoAttack.py                         Main genetic attack
GoAttackCoherence.py             Genetic attack with LLM-judged coherence loss
NoAttack.py                         Clean inference baseline
QuestionModification.py             Alternative-question generation
GetSolutionofQuestionModified.py    Alternative reasoning generation
Evaluation.py                       ACC/ASR/SS and efficiency metrics
CoherenceEvaluator.py               LLM-judge coherence evaluation
GeneticOptimizer.py                 Genetic search loop
FitnessEvaluator.py                 Composite fitness function
GeneticOperators.py                 Crossover and mutation operators
data/                                Evaluation datasets
models--sentence-transformers--all-MiniLM-L6-v2/
                                     Optional local sentence-encoder cache
model_utils.py                       Model download and cache utility
```

For anonymous review, this README intentionally contains no author identity or non-anonymous citation information.
