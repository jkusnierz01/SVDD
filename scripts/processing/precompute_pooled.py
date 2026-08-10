"""
Precompute pooled fp16 tensors from wav2vec + mert embeddings.

Input per file:
  preprocessed_16k_mono/wav2vec/{stem}.npy  -> [6000, 1024] float32
  preprocessed_16k_mono/mert/{stem}.npy     -> [6000, 1024] float32

Output per file:
  {output_dir}/{stem}.npy                   -> [1500, 2048] float16

Also writes {output_dir}/index.json with entries: {stem, pooled}.

Usage:
  python precompute_pooled.py \
      --index /path/to/preprocessed_16k_mono/index.json \
      --output_dir /path/to/preprocessed_pooled_fp16 \
      --workers 16
"""

import json
import argparse
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from multiprocessing import Pool


def _worker_init():
    global _pool_op
    _pool_op = nn.AvgPool1d(kernel_size=4, stride=4)


def _process_item(args):
    item, output_dir = args
    stem = item["stem"]
    out_path = Path(output_dir) / f"{stem}.npy"

    if out_path.exists():
        return stem, "skip"

    try:
        w2v = torch.from_numpy(np.load(item["wav2vec"])).float()   # [6000, 1024]
        mert = torch.from_numpy(np.load(item["mert"])).float()     # [6000, 1024]
    except Exception as e:
        return stem, f"load_error: {e}"

    # Concat -> [6000, 2048]
    cat = torch.cat([w2v, mert], dim=1)

    # AvgPool1d expects [batch, channels, length] -> transpose first
    pooled = _pool_op(cat.T.unsqueeze(0)).squeeze(0).T   # [1500, 2048]

    try:
        np.save(str(out_path), pooled.half().numpy())
    except Exception as e:
        return stem, f"save_error: {e}"

    return stem, "ok"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index",      required=True, help="Path to preprocessed_16k_mono/index.json")
    parser.add_argument("--output_dir", required=True, help="Output directory for pooled fp16 tensors")
    parser.add_argument("--workers",    type=int, default=8)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(args.index) as f:
        data = json.load(f)

    print(f"Total entries: {len(data)}")
    print(f"Output dir:    {output_dir}")
    print(f"Workers:       {args.workers}")
    print()

    stem_to_item = {item["stem"]: item for item in data}
    tasks = [(item, str(output_dir)) for item in data]

    new_index = []
    errors = []
    skipped = 0
    done = 0

    with Pool(processes=args.workers, initializer=_worker_init) as pool:
        for i, (stem, status) in enumerate(pool.imap_unordered(_process_item, tasks), 1):
            if status == "ok":
                done += 1
            elif status == "skip":
                skipped += 1
            else:
                errors.append((stem, status))

            out_path = output_dir / f"{stem}.npy"
            if out_path.exists():
                original = stem_to_item.get(stem, {})
                entry = {k: v for k, v in original.items() if k not in ("wav2vec", "mert")}
                entry["pooled"] = str(out_path)
                new_index.append(entry)

            if i % 1000 == 0 or i == len(data):
                print(f"[{i}/{len(data)}]  done={done}  skipped={skipped}  errors={len(errors)}")

    # Write new index
    index_out = output_dir / "index.json"
    with open(str(index_out), "w") as f:
        json.dump(new_index, f)

    print()
    print(f"Finished. Processed={done}, skipped={skipped}, errors={len(errors)}")
    if errors:
        print("First 20 errors:")
        for stem, msg in errors[:20]:
            print(f"  {stem}: {msg}")
    print(f"Index written to: {index_out}")


if __name__ == "__main__":
    main()
