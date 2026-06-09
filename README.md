# Generalizing Verifiable Instruction Following

This repo contains IFBench, which is a new, challenging benchmark for precise instruction following.
Read the <a href="https://arxiv.org/pdf/2507.02833">IFBench paper</a>, accepted to NeurIPS 2025, D&B.

## IFBench
IFBench consists of two parts:

- OOD Constraints: 58 new and challenging constraints, with corresponding verification functions. The constraint templates are combined with prompts from a held-out set of WildChat (Zhao et al. 2024).

- (optionally) Multiturn Constraint Isolation in 2 turns: The prompt and the constraint are separated over two turns, i.e. the first turn is the user prompt and the model's response to the prompt, and the second turn is the constraint that modifies the initial prompt.

- New IF-RLVR training constraints: 29 new and challenging constraints, with corresponding verification functions. 

## How to run generation & evaluation

Install dependencies first:
```
pip install -r requirements.txt
```

### Code layout (updated)

- `run_eval.py`: main entrypoint for online generation and evaluation.
- `src/ifbench_parquet.py`: shared parquet loader used by both online and batch generation.
- `src/evaluation_lib.py`: strict/loose evaluation and report writing.
- `src/instructions.py`, `src/instructions_registry.py`, `src/instructions_util.py`: instruction checkers and registry.
- `batch_api/batch.py`: batch pipeline for Moonshot/Kimi and Qwen (DashScope) models.

### Model configuration (`models.yaml`)

`run_eval.py` and batch scripts both read model config from `models.yaml`:
- `name`: model id passed via `--model-id`
- `base_url`: OpenAI-compatible endpoint
- `api_key`: supports env var form like `${DASHSCOPE_API_KEY}`
- optional `temperature`, `max_tokens`, `extra_body`

### Online generation (`run_eval.py`)

Run generation against IFBench parquet (expects `key`, `prompt`, `messages` or `message`):
```
python run_eval.py \
  --model-id deepseek-v4-pro \
  --input-parquet data/test-00000-of-00001.parquet \
  --save-to generation/deepseek-v4-pro/20260429_120000.jsonl
```

Useful flags:
- `--num-tasks`: only generate first N rows from parquet.
- `--max-workers`: thread count for generation (default `50`).
- `--models_yaml`: alternative model config path.
- `--save-to` omitted -> defaults to `generation/<model>/<timestamp>.jsonl`.

Generation output JSONL contains at least:
- `key`
- `prompt`
- `response`
- `total_tokens`

If `--save-to` already exists, generation auto-resumes by skipping existing `key`s.

### Evaluation (`run_eval.py`)

Evaluate with parquet + generation JSONL:
```
python run_eval.py \
  --eval-file generation/deepseek-v4-pro/20260429_120000.jsonl \
  --input-parquet data/test-00000-of-00001.parquet
```

Output defaults to `eval/<model>/<timestamp>/` (or set `--output-dir`), and writes:
- `strict.jsonl`
- `loose.jsonl`

Each file starts with a summary record, followed by per-example results.  
In the paper, we generally report prompt-level loose accuracy.

### Batch generation (Moonshot / Qwen)

`batch_api/batch.py` produces the same generation JSONL format as `run_eval.py`, so you can evaluate with the same evaluation command. Provider rules are inferred from `--model-id` (`kimi-*` → Moonshot, `qwen*` → Qwen).

```
python batch_api/batch.py \
  --model-id kimi-k2.6 \
  --input-parquet data/test-00000-of-00001.parquet \
  --save-to generation/kimi-k2.6/20260429_120000.jsonl \
  --step all
```

```
python batch_api/batch.py \
  --model-id qwen3.6-flash \
  --input-parquet data/test-00000-of-00001.parquet \
  --save-to generation/qwen3.6-flash/20260429_120000.jsonl \
  --step all
```

Defaults and provider notes:
- `--model-id` is required.
- artifacts under `batch_api/artifacts/<model>/<timestamp>/`
- `--max-tasks-per-batch` defaults to `1000` (set higher for Qwen if needed, e.g. `5000`)
- `--completion-window` must be in `[24h, 336h]`
- Kimi batch requests omit `temperature`, `max_tokens`, and related sampling params

#### Step-by-step mode

You can run stages manually:
1. `--step prepare` (build `batch_input_cXXX.jsonl`, `meta.json`, and key payload maps)
2. `--step upload --run-dir <run_dir>`
3. `--step create --run-dir <run_dir>`
4. `--step wait --run-dir <run_dir>`
5. `--step collect --run-dir <run_dir>`

For multi-chunk runs, use `--chunk-index` for split steps, or just run `--step all` to process every chunk sequentially.

## Released Datasets
You can find our released datasets in this [collection](https://huggingface.co/collections/allenai/ifbench-683f590687f61b512558cdf1), which contains the [test data](https://huggingface.co/datasets/allenai/IFBench_test), the [multi-turn test data](https://huggingface.co/datasets/allenai/IFBench_multi-turn) and the [IF-RLVR training data](https://huggingface.co/datasets/allenai/IF_multi_constraints_upto5).

## RLVR for Precise Instruction Following
We also release our IF-RLVR code, as part of [open-instruct](https://github.com/allenai/open-instruct). You can run this [GRPO script](https://github.com/allenai/open-instruct/blob/main/open_instruct/grpo_fast.py), using our [training data](https://huggingface.co/datasets/allenai/IF_multi_constraints_upto5). This is an [example command](https://github.com/allenai/open-instruct/blob/main/scripts/train/rlvr/valpy_if_grpo_fast.sh).

The new training constraints and verification functions are here: https://github.com/allenai/open-instruct/tree/main/open_instruct/IFEvalG

## 📊 Model Performance Leaderboard

| Rank | Model | IFBench Score | IFEval Score |
|------|-------|---------------|--------------|
| 🥇 1 | OpenAI o3 | **69.3** | 95.0 |
| 🥈 2 | Qwen2.5 Base + IF-RLVR | **53.7** | 87.8 |
| 🥉 3 |  Llama 3.1 Base + IF-RLVR | **52.7** | 88.2 |
| 4 | Gemini 2.5 Pro | 52.3 | 65.4 |
| 5 | Qwen 2.5 Instruct + IF-RLVR | 48.7 | 89.1 |
| 6 | OLMo2 Base + IF-RLVR | 47.3 | 70.4 |
| 7 | OLMo2 Instruct + IF-RLVR | 44.7 | 74.5 |
| 7 | Tulu3 DPO + IF-RLVR | 43.3 | 92.2 |
| 9 | Claude 4 Sonnet | 42.3 | 91.3 |
| 10 | DeepSeek R1 | 38.0 | 86.13 |
| 11 | Qwen 3 32B | 37.3 | 85.6 |
| 12 | Qwen 3 8B | 35.0 | 86.3 |

*Sorted by IFBench score (higher is better)*
If you want your model added to the leaderboard, please create a pull request or email me!

## Licensing

This codebase is licensed under Apache 2.0 as given in [LICENSE](./LICENSE).

The data is licensed under ODC-BY-1.0. It is intended for research and educational use in accordance with Ai2's Responsible Use Guidelines. The dataset includes output data generated from third party models that are subject to separate terms governing their use.


## Acknowledgements

Parts of IFBench are built upon and extend [IFEval](https://github.com/google-research/google-research/tree/master/instruction_following_eval) (Zhou et al. 2023) and we would like to thank them for their great work!


## Citation

If you used this repository or our models, please cite our work:

```bibtex
@misc{pyatkin2025generalizing,
   title={Generalizing Verifiable Instruction Following}, 
   author={Valentina Pyatkin and Saumya Malik and Victoria Graf and Hamish Ivison and Shengyi Huang and Pradeep Dasigi and Nathan Lambert and Hannaneh Hajishirzi},
   year={2025},
  journal={Advances in Neural Information Processing Systems},
  volume={38},
  year={2025}
}
