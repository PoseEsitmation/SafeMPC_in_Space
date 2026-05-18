import sys
import argparse
import hypercrl
import torch


# ----------------------------
# DEVICE PARSING (central)
# ----------------------------
def parse_device(device_str: str):

    if device_str is None:
        return (
            "cuda:0" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )

    if device_str.startswith("cuda"):
        if torch.cuda.is_available():
            return device_str
        raise RuntimeError("CUDA not available, falling back to CPU.")

    if device_str == "mps":
        if torch.backends.mps.is_available():
            return "mps"
        raise RuntimeError("MPS not available, falling back to CPU.")

    return "cpu"

# ----------------------------
# WRAPPERS
# ----------------------------
def run_hnet(args):
    return hypercrl.hnet(
        args.env,
        seed=args.seed,
        savepath=args.savepath,
        play=args.play,
        device=args.device
    )


def run_pnn(args):
    return hypercrl.pnn(args.env)


def run_ewc(args):
    return hypercrl.ewc(args.env)


# ----------------------------
# CLI
# ----------------------------
def main():
    parser = argparse.ArgumentParser()

    subparsers = parser.add_subparsers(dest="cmd", required=True)

    # ---------------- HNET ----------------
    hnet_parser = subparsers.add_parser("hnet")
    hnet_parser.add_argument("env")
    hnet_parser.add_argument("--seed", type=int, default=None)
    hnet_parser.add_argument("--savepath", type=str, default=None)
    hnet_parser.add_argument("--play", action="store_true")
    hnet_parser.add_argument("--device", type=str, default=None)

    # ---------------- PNN ----------------
    pnn_parser = subparsers.add_parser("pnn")
    pnn_parser.add_argument("env")

    # ---------------- EWC ----------------
    ewc_parser = subparsers.add_parser("ewc")
    ewc_parser.add_argument("env")

    args = parser.parse_args()

    # device resolution
    if hasattr(args, "device"):
        args.device = parse_device(args.device)

    print(f"[DEBUG] Using device: {getattr(args, 'device', None)}")

    # dispatch
    if args.cmd == "hnet":
        run_hnet(args)
    elif args.cmd == "pnn":
        run_pnn(args)
    elif args.cmd == "ewc":
        run_ewc(args)


if __name__ == "__main__":
    main()