import json
import logging as py_logging
import os
import sys
import threading
import time
from typing import Any, Dict, Iterable, List, Optional, Set
from datetime import datetime
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from absl import app
from absl import flags
from absl import logging
import evaluation_lib
import pandas as pd
import yaml
from openai import OpenAI
from tqdm import tqdm


_HARDCODED_MODEL_ID = "MiniMax-M2-Stable"

_GENERATION_MAX_WORKERS = 30
_GENERATION_MAX_RETRIES = 3

_MODELS_YAML = flags.DEFINE_string(
    "models_yaml", "models.yaml", "Path to models.yaml.", required=False
)

_INPUT_PARQUET = flags.DEFINE_string(
    "input_parquet",
    os.path.join("data", "test-00000-of-00001.parquet"),
    "Path to IFBench parquet containing the chat messages.",
    required=False,
)

_NUM_TASKS = flags.DEFINE_integer(
    "num_tasks",
    None,
    "If provided, only run the first k tasks from --input_parquet (generation only).",
    required=False,
)

_OUTPUT_FILE = flags.DEFINE_string(
    "output_file",
    None,
    "JSONL file to write generation outputs. If not provided, defaults to ./output/<model>.jsonl",
    required=False,
)

_INPUT_DATA = flags.DEFINE_string(
    "input_data",
    None,
    "Path to model generation JSONL (must contain prompt/response; used for evaluation).",
    required=False,
)

_OUTPUT_DIR = flags.DEFINE_string(
    "output_dir",
    "eval",
    "Output directory for evaluation results.",
    required=False,
)


if hasattr(flags, "DEFINE_alias"):
  flags.DEFINE_alias("output-file", "output_file")
  flags.DEFINE_alias("input-data", "input_data")
  flags.DEFINE_alias("models-yaml", "models_yaml")
  flags.DEFINE_alias("input-parquet", "input_parquet")
  flags.DEFINE_alias("num-tasks", "num_tasks")


def _resolve_env_vars(value: Any) -> Any:
  if isinstance(value, str):
    if value.startswith("${") and value.endswith("}") and len(value) >= 4:
      env_key = value[2:-1]
      return os.environ.get(env_key)
    return value
  if isinstance(value, dict):
    return {k: _resolve_env_vars(v) for k, v in value.items()}
  if isinstance(value, list):
    return [_resolve_env_vars(v) for v in value]
  return value


def _load_model_config(models_yaml_path: str, model_id: str) -> Dict[str, Any]:
  with open(models_yaml_path, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

  if not isinstance(cfg, dict) or "models" not in cfg or not isinstance(cfg["models"], list):
    raise ValueError("Invalid models.yaml: expected top-level key 'models' as a list.")

  for m in cfg["models"]:
    if isinstance(m, dict) and m.get("name") == model_id:
      resolved = _resolve_env_vars(m)
      if not resolved.get("api_key"):
        raise ValueError(
            f"Model '{model_id}' is missing api_key (after env var resolution)."
        )
      if not resolved.get("base_url"):
        raise ValueError(f"Model '{model_id}' is missing base_url.")
      return resolved
  raise ValueError(f"Unknown --model-id: '{model_id}' (not found in {models_yaml_path}).")


def _read_existing_keys(output_file: str) -> Set[int]:
  existing_keys: Set[int] = set()
  if not os.path.exists(output_file):
    return existing_keys

  with open(output_file, "r", encoding="utf-8") as f:
    for line_num, line in enumerate(f, start=1):
      line = line.strip()
      if not line:
        continue
      try:
        obj = json.loads(line)
      except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in {output_file} at line {line_num}: {e}") from e
      if "key" not in obj:
        raise ValueError(
            f"Missing 'key' in existing output file {output_file} at line {line_num}."
        )
      k = obj["key"]
      if not isinstance(k, int):
        raise ValueError(
            f"Non-int 'key' in existing output file {output_file} at line {line_num}."
        )
      existing_keys.add(k)

  return existing_keys


def _ensure_parent_dir(path: str) -> None:
  parent = os.path.dirname(path)
  if parent:
    os.makedirs(parent, exist_ok=True)


def _read_benchmark_inputs_from_parquet(parquet_path: str) -> List[evaluation_lib.InputExample]:
  df = pd.read_parquet(parquet_path)
  required_cols = {"key", "prompt", "instruction_id_list", "kwargs"}
  missing = required_cols - set(df.columns)
  if missing:
    raise ValueError(f"Parquet is missing required columns: {sorted(missing)}")

  def _normalize_kwargs_list(kwargs_list: Any) -> list[Dict[str, Optional[Any]]]:
    normalized: list[Dict[str, Optional[Any]]] = []
    for item in list(kwargs_list):
      if item is None:
        normalized.append({})
        continue
      if not isinstance(item, dict):
        raise ValueError("Parquet 'kwargs' items must be dicts.")
      new_item: Dict[str, Optional[Any]] = {}
      for k, v in item.items():
        if v is None:
          new_item[k] = None
        elif isinstance(v, float) and v.is_integer():
          new_item[k] = int(v)
        else:
          new_item[k] = v
      normalized.append(new_item)
    return normalized

  inputs: List[evaluation_lib.InputExample] = []
  for row in df.itertuples(index=False):
    inputs.append(
        evaluation_lib.InputExample(
            key=int(getattr(row, "key")),
            instruction_id_list=list(getattr(row, "instruction_id_list")),
            prompt=str(getattr(row, "prompt")),
            kwargs=_normalize_kwargs_list(getattr(row, "kwargs")),
        )
    )
  return inputs


def _read_prompt_to_response_dict_from_generations(generations_jsonl_path: str) -> Dict[str, str]:
  prompt_to_response: Dict[str, str] = {}
  with open(generations_jsonl_path, "r", encoding="utf-8") as f:
    for line_num, line in enumerate(f, start=1):
      line = line.strip()
      if not line:
        continue
      try:
        ex = json.loads(line)
      except json.JSONDecodeError as e:
        raise ValueError(
            f"Invalid JSON in {generations_jsonl_path} at line {line_num}: {e}"
        ) from e
      if "prompt" not in ex or "response" not in ex:
        raise ValueError(
            f"Each line must contain 'prompt' and 'response' (line {line_num})."
        )
      prompt = ex["prompt"]
      response = ex["response"]
      if not isinstance(prompt, str) or not isinstance(response, str):
        raise ValueError(
            f"'prompt' and 'response' must be strings (line {line_num})."
        )
      if prompt in prompt_to_response:
        raise ValueError(
            f"Duplicate prompt found in {generations_jsonl_path} at line {line_num}."
        )
      prompt_to_response[prompt] = response
  return prompt_to_response


def _get_num_tasks_or_none() -> Optional[int]:
  if _NUM_TASKS.value is None:
    return None
  if _NUM_TASKS.value <= 0:
    raise ValueError("--num_tasks must be a positive integer.")
  return int(_NUM_TASKS.value)


def _run_generation() -> None:
  model_cfg = _load_model_config(_MODELS_YAML.value, _HARDCODED_MODEL_ID)

  model_name = model_cfg["name"]
  temperature = model_cfg.get("temperature")
  base_url = model_cfg["base_url"]
  api_key = model_cfg["api_key"]
  extra_body = model_cfg.get("extra_body")
  max_tokens=model_cfg.get("max_tokens")

  py_logging.getLogger("openai").setLevel(py_logging.WARNING)
  py_logging.getLogger("openai._client").setLevel(py_logging.WARNING)
  py_logging.getLogger("httpx").setLevel(py_logging.WARNING)
  py_logging.getLogger("httpcore").setLevel(py_logging.WARNING)

  output_file = _OUTPUT_FILE.value
  if not output_file:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join("output", f"{_HARDCODED_MODEL_ID}_{ts}.jsonl")

  _ensure_parent_dir(output_file)
  existing_keys = _read_existing_keys(output_file)
  if existing_keys:
    logging.info("Resuming generation: %d keys already in %s", len(existing_keys), output_file)

  df = pd.read_parquet(_INPUT_PARQUET.value)
  num_tasks = _get_num_tasks_or_none()
  if num_tasks is not None:
    df = df.head(num_tasks)
  if "messages" not in df.columns and "message" not in df.columns:
    raise ValueError("Parquet must contain 'messages' column (OpenAI chat format).")
  if "messages" not in df.columns:
    df = df.rename(columns={"message": "messages"})
  if "key" not in df.columns or "prompt" not in df.columns:
    raise ValueError("Parquet must contain 'key' and 'prompt' columns.")

  def _make_create_kwargs(messages: List[Dict[str, Any]]) -> Dict[str, Any]:
    create_kwargs: Dict[str, Any] = {
        "model": model_name,
        "messages": messages,
    }
    if temperature is not None:
      create_kwargs["temperature"] = float(temperature)
    if extra_body is not None:
      create_kwargs["extra_body"] = extra_body
    if max_tokens is not None:
      create_kwargs["max_tokens"] =max_tokens
    return create_kwargs

  thread_local = threading.local()

  def _get_client() -> OpenAI:
    client = getattr(thread_local, "client", None)
    if client is None:
      client = OpenAI(base_url=base_url, api_key=api_key)
      thread_local.client = client
    return client

  def _stream_chat_completion_text(client: OpenAI, create_kwargs: Dict[str, Any]) -> str:
    parts: List[str] = []
    stream_kwargs = dict(create_kwargs)
    stream_kwargs["stream"] = True

    for chunk in client.chat.completions.create(**stream_kwargs):
      choices = getattr(chunk, "choices", None)
      if not choices:
        continue
      choice0 = choices[0]
      delta = getattr(choice0, "delta", None)
      if delta is None and isinstance(choice0, dict):
        delta = choice0.get("delta")
      if delta is None:
        continue

      content_piece = getattr(delta, "content", None)
      if content_piece is None and isinstance(delta, dict):
        content_piece = delta.get("content")
      if isinstance(content_piece, str) and content_piece:
        parts.append(content_piece)

    return "".join(parts)

  def _generate_one(key: int, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
    client = _get_client()
    create_kwargs = _make_create_kwargs(messages)

    last_exc: Optional[Exception] = None
    for attempt in range(_GENERATION_MAX_RETRIES):
      try:
        content = _stream_chat_completion_text(client, create_kwargs)
        if content == "":
          resp = client.chat.completions.create(**create_kwargs)
          content = resp.choices[0].message.content
          if content is None:
            content = ""
        return {"key": key, "response": content}
      except Exception as e:
        if _is_sensitive_content_error(e):
          raise
        last_exc = e
        if attempt < _GENERATION_MAX_RETRIES - 1:
          sleep_s = min(8.0, float(2 ** attempt))
          logging.warning(
              "Non-sensitive generation error for key=%s (attempt %d/%d): %s; retrying in %.1fs",
              key,
              attempt + 1,
              _GENERATION_MAX_RETRIES,
              e,
              sleep_s,
          )
          time.sleep(sleep_s)
        else:
          logging.warning(
              "Non-sensitive generation error for key=%s (attempt %d/%d): %s; giving up",
              key,
              attempt + 1,
              _GENERATION_MAX_RETRIES,
              e,
          )

    assert last_exc is not None
    raise last_exc

  def _is_sensitive_content_error(e: Exception) -> bool:
    text = str(e)
    if "敏感" in text or "contentFilter" in text:
      return True
    body = getattr(e, "body", None)
    if isinstance(body, dict):
      try:
        body_text = json.dumps(body, ensure_ascii=False)
      except Exception:
        body_text = str(body)
      if "敏感" in body_text or "contentFilter" in body_text:
        return True
      err_obj = body.get("error")
      if isinstance(err_obj, dict):
        msg = err_obj.get("message")
        if isinstance(msg, str) and "敏感" in msg:
          return True
      return False
    if isinstance(body, str) and "敏感" in body:
      return True
    return False

  def _prompt_skip_sensitive_task(key: int, e: Exception) -> bool:
    if hasattr(sys.stdin, "isatty") and not sys.stdin.isatty():
      return False
    print("\n" + "=" * 80)
    print(f"Task key={key} failed: possible sensitive content in request/response.")
    print(f"Error: {e}")
    while True:
      ans = input("Skip this task and continue? [y/N]: ").strip().lower()
      if ans in ("y", "yes"):
        return True
      if ans in ("", "n", "no"):
        return False

  pending: List[tuple[int, List[Dict[str, Any]]]] = []
  prompt_by_key: Dict[int, str] = {}
  done_count = 0
  for row in df.itertuples(index=False):
    key = int(getattr(row, "key"))
    if key in existing_keys:
      done_count += 1
      continue
    prompt = str(getattr(row, "prompt"))
    prompt_by_key[key] = prompt
    raw_messages = getattr(row, "messages")
    messages = list(raw_messages)
    pending.append((key, messages))

  total_count = len(df)
  pbar = tqdm(total=total_count, initial=done_count, desc="Generating", unit="task")
  try:
    with open(output_file, "a", encoding="utf-8") as out_f:
      with ThreadPoolExecutor(max_workers=_GENERATION_MAX_WORKERS) as executor:
        futures: List[Future] = []
        future_to_key: Dict[Future, int] = {}
        for key, messages in pending:
          fut = executor.submit(_generate_one, key, messages)
          futures.append(fut)
          future_to_key[fut] = key

        for fut in as_completed(futures):
          try:
            record = fut.result()
          except Exception as e:
            key = future_to_key.get(fut)
            if key is not None and _is_sensitive_content_error(e):
              if _prompt_skip_sensitive_task(key, e):
                pbar.update(1)
                continue
            for other in futures:
              other.cancel()
            logging.exception("Generation failed")
            raise e

          key = int(record["key"])
          record["prompt"] = prompt_by_key.get(key, "")

          response = record.get("response")
          if response is None or (isinstance(response, str) and response.strip() == ""):
            print(f"EMPTY_RESPONSE key={key}")
            pbar.update(1)
            continue

          out_f.write(json.dumps(record, ensure_ascii=False))
          out_f.write("\n")
          out_f.flush()
          existing_keys.add(int(record["key"]))
          pbar.update(1)
  finally:
    pbar.close()

  logging.info("Generation completed. Output: %s", output_file)


def _run_evaluation() -> None:
  inputs = _read_benchmark_inputs_from_parquet(_INPUT_PARQUET.value)
  prompt_to_response = _read_prompt_to_response_dict_from_generations(_INPUT_DATA.value)

  output_dir = _OUTPUT_DIR.value
  os.makedirs(output_dir, exist_ok=True)

  input_base = os.path.splitext(os.path.basename(_INPUT_DATA.value))[0]
  ts = datetime.now().strftime("%Y%m%d_%H%M%S")

  for func, output_file_name in [
      (evaluation_lib.test_instruction_following_strict, "eval_results_strict"),
      (evaluation_lib.test_instruction_following_loose, "eval_results_loose"),
  ]:
    suffix = "strict" if func == evaluation_lib.test_instruction_following_strict else "loose"
    logging.info("Generating %s...", output_file_name)
    outputs = []
    missing_prompts = 0
    for inp in inputs:
      if inp.prompt not in prompt_to_response:
        missing_prompts += 1
        continue
      outputs.append(func(inp, prompt_to_response))

    evaluated = len(outputs)
    total = len(inputs)
    if evaluated == 0:
      raise ValueError(
          "No prompts in --input-data matched the benchmark prompts; cannot evaluate."
      )
    if missing_prompts:
      logging.warning(
          "Partial evaluation: evaluated %d/%d prompts (missing %d).",
          evaluated,
          total,
          missing_prompts,
      )

    follow_all_instructions = [o.follow_all_instructions for o in outputs]
    accuracy = sum(follow_all_instructions) / len(outputs)
    logging.info("Accuracy: %f", accuracy)

    output_file_name = os.path.join(output_dir, f"{input_base}_{suffix}.jsonl")
    report_dict = evaluation_lib.compute_report(outputs)
    summary_record = {
        "type": "summary",
        "mode": suffix,
        "evaluated": evaluated,
        "total": total,
        "missing": missing_prompts,
        **report_dict,
    }

    evaluation_lib.write_outputs_with_prefix(output_file_name, outputs, summary_record)

    logging.info("Generated: %s", output_file_name)

    print("=" * 64)
    print(f"{output_file_name} Accuracy Scores:")
    evaluation_lib.print_report(outputs)


def main(argv):
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  if _INPUT_DATA.value:
    _run_evaluation()
    return

  _run_generation()


if __name__ == "__main__":
  app.run(main)
