import argparse
import os
import json

from core import Sequence, Dataset


def parse_args():
    parser = argparse.ArgumentParser(description="Point Cloud Overlapping Patch Sampler")
    parser.add_argument(
        "-d",
        "--data-root",
        type=str,
        default="data-set",
        help="Data Root Directory",
        metavar="DIR"
    )
    parser.add_argument(
        "-o",
        "--out-dir",
        type=str,
        default="out",
        help="Output Directory",
        metavar="OUT-DIR"
    )
    parser.add_argument(
        "-c",
        "--config-file",
        type=str,
        default="data-config.json",
        help="Dataset Configuration File",
        metavar="data-config.json"
    )
    args = parser.parse_args()
    with open(os.path.join(args.data_root, args.config_file), 'r') as conf:
        config = json.load(conf)

    return args, config


def parse_sequence(args, name, seq):
    filename = seq["filename"]
    transform = seq["transform"] if "transform" in seq else None
    return Sequence(args.data_root, name, filename, transform)


def parse_dataset(args, seq_map, dset):
    return Dataset(
        out_dir=args.out_dir,
        name=dset["name"],
        first_seq=seq_map[dset["first"]],
        second_seq=seq_map[dset["second"]],
        rect_size=dset["rectangle-size"],
        interval=dset["interval"],
        min_percent=dset["min-percent"],
        normalize_output=dset["normalize-output"] if "normalize-output" in dset else False
    )


def main():
    args, config = parse_args()
    sequences = {name: parse_sequence(args, name, seq) for name, seq in config["seqs"].items()}
    for dset in config["datasets"]:
        dataset = parse_dataset(args, sequences, dset)
        print("Processing Dataset", dataset.name)
        print(dataset)
        print("=====")
        dataset.process()
        print("Processing finished\n")


if __name__ == "__main__":
    main()
