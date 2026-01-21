# main.py
import argparse
from pathlib import Path
import yaml

from src.embedding.run_embedding_comparison import run_embedding_pipeline

ROOT = Path(__file__).resolve().parent

def load_yaml(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["embedding"], required=True)
    ap.add_argument("--config", default="configs/embedding.yaml")
    ap.add_argument("--platform", required=True, help="Amazon | Hotel | Coursera | Audible ...")
    args = ap.parse_args()

    cfg = load_yaml(ROOT / args.config)

    if args.task == "embedding":
        run_embedding_pipeline(
            cfg=cfg,
            platform=args.platform,
            project_root=ROOT
        )

if __name__ == "__main__":
    main()