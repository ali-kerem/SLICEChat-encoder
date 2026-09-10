import os
import sys
from tqdm import tqdm

import pandas as pd
import torch

from .data import get_data
from .params import parse_args
from .train import evaluate
from .main import random_seed
from ..open_clip.factory import create_customtextclip_and_tokenizer_from_ckpt


def main(args):
    args = parse_args(args)

    args.distributed = False
    args.rank = 0
    args.save_logs = False
    args.wandb = False

    if torch.cuda.is_available():
        # This enables tf32 on Ampere GPUs which is only 8% slower than
        # float16 and almost as accurate as float32
        # This was a default in pytorch until 1.12
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    random_seed(args.seed, 0)
    if os.path.exists(args.eval_results_save_path):
        results_df = pd.read_csv(args.eval_results_save_path)
    else:
        results_df = pd.DataFrame()

    for ckpt_path in tqdm(args.eval_ckpt_paths, desc="Evaluating models"):
        log_dir = "/".join(ckpt_path.rstrip("/").split("/")[:-2])
        run_name = log_dir.split("/")[-1]

        model, tokenizer = create_customtextclip_and_tokenizer_from_ckpt(ckpt_path, device=torch.device("cuda"), dtype=args.precision)

        random_seed(args.seed, 0)

        data = get_data(
            args,
            (None, None),
            epoch=0,
            tokenizer=tokenizer,
        )
        assert len(data), 'At least eval dataset must be specified.'

        metrics = evaluate(model, data, 0, args, tokenizer=tokenizer)

        del metrics["epoch"]
        del metrics["num_samples"]

        results_df = pd.concat([results_df, pd.DataFrame([{"run": run_name, **metrics}])], ignore_index=True)
        results_df.to_csv(args.eval_results_save_path, index=False)


if __name__ == "__main__":
    main(sys.argv[1:])
