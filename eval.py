import os
import sys
import argparse

import vaetc
from vis import main as visualize_gwae
from clu import main as visualize_cluster
sys.path.append("./")
import models
sys.path.pop()

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("logger_path", type=str)
    parser.add_argument("--no-evaluate", action="store_true")
    parser.add_argument("--no-quant", action="store_true")
    parser.add_argument("--no-qual", action="store_true")
    parser.add_argument("--no-gwae", action="store_true")
    parser.add_argument("--no-cluster", action="store_true")
    args = parser.parse_args()

    cp = vaetc.load_checkpoint(os.path.join(args.logger_path, "checkpoint_last.pth"))
    if not args.no_evaluate:
        vaetc.evaluate(cp, qualitative=not args.no_qual, quantitative=not args.no_quant)
    if not args.no_gwae:
        visualize_gwae(cp)
    if not args.no_cluster:
        visualize_cluster(cp)