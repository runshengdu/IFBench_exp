import argparse
import json
import os
from typing import Any, Dict, List, Optional

import pandas as pd
import yaml

try:
  from openai import OpenAI
except ModuleNotFoundError as e:  # pragma: no cover
  raise ModuleNotFoundError(
      "Missing dependency 'openai'. Please install it (e.g. pip install -r requirements.txt) and retry."
  ) from e


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


def _load_model_config(models_yaml_path: str, model_name: str) -> Dict[str, Any]:
  with open(models_yaml_path, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

  if not isinstance(cfg, dict) or "models" not in cfg or not isinstance(cfg["models"], list):
    raise ValueError("Invalid models.yaml: expected top-level key 'models' as a list.")

  for m in cfg["models"]:
    if not isinstance(m, dict):
      continue
    if m.get("name") == model_name:
      resolved = _resolve_env_vars(m)
      if not resolved.get("api_key"):
        raise ValueError(
            f"API key for model '{model_name}' is missing after env var resolution. "
            "Check your environment variables."
        )
      return resolved

  available = [m.get("name") for m in cfg["models"] if isinstance(m, dict)]
  raise ValueError(f"Model '{model_name}' not found in {models_yaml_path}. Available: {available}")


def _first_empty_response_keys(generations_jsonl_path: str, limit: int) -> List[int]:
  keys: List[int] = []
  with open(generations_jsonl_path, "r", encoding="utf-8") as f:
    for line in f:
      if not line.strip():
        continue
      try:
        obj = json.loads(line)
      except Exception:
        continue
      if obj.get("response") == "":
        k = obj.get("key")
        if k is None:
          continue
        try:
          keys.append(int(k))
        except Exception:
          continue
        if len(keys) >= limit:
          break
  return keys


def _messages_by_key_from_parquet(input_parquet: str, keys: List[int]) -> Dict[int, List[Dict[str, Any]]]:
  df = pd.read_parquet(input_parquet)
  if "messages" not in df.columns and "message" not in df.columns:
    raise ValueError("Parquet must contain 'messages' column (OpenAI chat format).")
  if "messages" not in df.columns:
    df = df.rename(columns={"message": "messages"})
  if "key" not in df.columns:
    raise ValueError("Parquet must contain 'key' column.")

  key_set = set(keys)
  sub = df[df["key"].astype(int).isin(key_set)]
  out: Dict[int, List[Dict[str, Any]]] = {}
  for row in sub.itertuples(index=False):
    k = int(getattr(row, "key"))
    raw_messages = getattr(row, "messages")
    out[k] = list(raw_messages)
  return out


def _safe_json_dumps(x: Any) -> str:
  try:
    return json.dumps(x, ensure_ascii=False)
  except Exception:
    return str(x)


def _extract_message_debug(choice_message: Any) -> Dict[str, Any]:
  debug: Dict[str, Any] = {}
  content = getattr(choice_message, "content", None)
  debug["content"] = content
  tool_calls = getattr(choice_message, "tool_calls", None)
  if tool_calls is not None:
    debug["tool_calls"] = tool_calls
  refusal = getattr(choice_message, "refusal", None)
  if refusal is not None:
    debug["refusal"] = refusal
  return debug


def _run_one(client: OpenAI, model_cfg: Dict[str, Any], key: int, messages: List[Dict[str, Any]]) -> None:
  model_name = model_cfg["name"]
  temperature = model_cfg.get("temperature")
  extra_body = model_cfg.get("extra_body")

  create_kwargs: Dict[str, Any] = {"model": model_name, "messages": messages}
  if temperature is not None:
    create_kwargs["temperature"] = float(temperature)
  if extra_body is not None:
    create_kwargs["extra_body"] = extra_body

  print("=" * 100)
  print(f"key={key}")
  print(f"messages_len={len(messages)}")

  try:
    finish_reason = None
    parts: List[str] = []
    stream_kwargs = dict(create_kwargs)
    stream_kwargs["stream"] = True
    for chunk in client.chat.completions.create(**stream_kwargs):
      choices = getattr(chunk, "choices", None)
      if not choices:
        continue
      choice0 = choices[0]
      chunk_finish_reason = getattr(choice0, "finish_reason", None)
      if chunk_finish_reason is None and isinstance(choice0, dict):
        chunk_finish_reason = choice0.get("finish_reason")
      if chunk_finish_reason is not None:
        finish_reason = chunk_finish_reason

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

    content = "".join(parts)

    msg_debug: Dict[str, Any]
    if content == "":
      resp = client.chat.completions.create(**create_kwargs)
      choice0 = resp.choices[0]
      finish_reason = getattr(choice0, "finish_reason", None)
      msg = getattr(choice0, "message", None)
      msg_debug = _extract_message_debug(msg) if msg is not None else {"message": None}

      content = None
      if msg is not None:
        content = getattr(msg, "content", None)
      if content is None:
        content = ""
    else:
      msg_debug = {"content": content}

    print(f"finish_reason={finish_reason}")
    print(f"prompt={messages}")
    print("message_debug=" + _safe_json_dumps(msg_debug))

    print("content_is_empty=" + str(content == ""))
    if content != "":
      preview = content if len(content) <= 800 else content[:800] + "..."
      print("content_preview:\n" + preview)
    else:
      print("content_preview: <EMPTY>")

  except Exception as e:
    print("ERROR_TYPE=" + type(e).__name__)
    print("ERROR_STR=" + str(e))
    body = getattr(e, "body", None)
    if body is not None:
      print("ERROR_BODY=" + _safe_json_dumps(body))


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument(
      "--generations-jsonl",
      default=os.path.join("output", "MiniMax-M2-Stable_20251219_184601.jsonl"),
  )
  parser.add_argument(
      "--input-parquet",
      default=os.path.join("data", "test-00000-of-00001.parquet"),
  )
  parser.add_argument("--models-yaml", default="models.yaml")
  parser.add_argument("--model-name", default="MiniMax-M2-Stable")
  parser.add_argument("--num", type=int, default=5)
  args = parser.parse_args()

  model_cfg = _load_model_config(args.models_yaml, args.model_name)
  base_url = model_cfg["base_url"]
  api_key = model_cfg["api_key"]

  keys = _first_empty_response_keys(args.generations_jsonl, limit=args.num)
  if not keys:
    raise SystemExit("No empty responses found in generations file.")

  messages_by_key = _messages_by_key_from_parquet(args.input_parquet, keys)

  missing = [k for k in keys if k not in messages_by_key]
  if missing:
    print("WARNING: keys missing from parquet: " + ",".join(map(str, missing)))

  client = OpenAI(base_url=base_url, api_key=api_key)

  for k in keys:
    msgs = messages_by_key.get(k)
    if not msgs:
      continue
    _run_one(client, model_cfg, k, msgs)


if __name__ == "__main__":
  main()
