import argparse

from musiccaps.train_grpo import main as grpo_main
from musiccaps.train_sft import main as sft_main


def _cli():
    p = argparse.ArgumentParser(prog="python -m musiccaps")
    sub = p.add_subparsers(dest="cmd", required=True)
    s_sft = sub.add_parser("sft", help="Supervised fine-tuning")
    s_sft.add_argument("--config", type=str, default="configs/default.yaml")
    s_grpo = sub.add_parser("grpo", help="GRPO after SFT")
    s_grpo.add_argument("--config", type=str, default="configs/default.yaml")
    args = p.parse_args()
    if args.cmd == "sft":
        sft_main(args.config)
    else:
        grpo_main(args.config)


if __name__ == "__main__":
    _cli()
