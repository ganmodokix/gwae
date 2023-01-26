import os
import argparse

import vaetc
import yaml
import models

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("settings_path", type=str, help="path to settings YAML")
    parser.add_argument("--proceed", "-p", action="store_true", help="continue, load existing weights")
    parser.add_argument("--no_training", "-n", action="store_true", help="only evaluation (using with -p)")
    parser.add_argument("--seed", "-s", type=int, default=None, help="deterministic if seed is specified")
    args = parser.parse_args()

    if args.seed is not None:
        vaetc.deterministic(args.seed)

    with open(args.settings_path, "r") as fp:
        options = yaml.safe_load(fp)
        options["hyperparameters"] = yaml.safe_dump(options["hyperparameters"])

    if not args.proceed:

        checkpoint = vaetc.Checkpoint(options)
        if not args.no_training:
            vaetc.fit(checkpoint)

    else:

        checkpoint = vaetc.load_checkpoint(os.path.join(options["logger_path"], "checkpoint_last.pth"))
        if not args.no_training:
            vaetc.proceed(checkpoint, extend_epochs=None)

    vaetc.evaluate(checkpoint, checkpoint)
