# coding=utf-8
"""Shared IFBench parquet loading for generation (run_eval + batch)."""

from __future__ import annotations

from typing import Optional

import pandas as pd


def normalize_num_tasks(num_tasks: Optional[int]) -> Optional[int]:
  """None = use all rows; otherwise a positive int row limit."""
  if num_tasks is None:
    return None
  if num_tasks <= 0:
    raise ValueError("--num-tasks must be a positive integer.")
  return int(num_tasks)


def load_ifbench_generation_dataframe(
    parquet_path: str,
    num_tasks: Optional[int] = None,
) -> pd.DataFrame:
  """Load IFBench table for chat generation: key, prompt, messages (or message)."""
  n = normalize_num_tasks(num_tasks)
  df = pd.read_parquet(parquet_path)
  if n is not None:
    df = df.head(n)
  if "messages" not in df.columns and "message" not in df.columns:
    raise ValueError(
        "Parquet must contain 'messages' column (OpenAI chat format), or 'message' as an alias."
    )
  if "messages" not in df.columns:
    df = df.rename(columns={"message": "messages"})
  if "key" not in df.columns or "prompt" not in df.columns:
    raise ValueError("Parquet must contain 'key' and 'prompt' columns.")
  return df
