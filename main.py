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

# --- Skeletons for future extension ---

def run_coreset(args):
    return hypercrl.coreset(args.env)


def run_finetune(args):
    return hypercrl.finetune(args.env)


def run_single(args):
    return hypercrl.single(args.env)


def run_si(args):
    return hypercrl.si(args.env)


def run_multitask(args):
    return hypercrl.multitask(args.env)


def run_chunked_hnet(args):
    return hypercrl.chunked_hnet(args.env)


def run_hnet_mt(args):
    return hypercrl.hnet_mt(args.env)


def run_hnet_replay(args):
    return hypercrl.hnet_replay(args.env)


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

    # --- Optional / future methods ---

    coreset_parser = subparsers.add_parser("coreset")
    coreset_parser.add_argument("env")

    finetune_parser = subparsers.add_parser("finetune")
    finetune_parser.add_argument("env")

    single_parser = subparsers.add_parser("single")
    single_parser.add_argument("env")

    si_parser = subparsers.add_parser("si")
    si_parser.add_argument("env")

    multitask_parser = subparsers.add_parser("multitask")
    multitask_parser.add_argument("env")

    chunked_parser = subparsers.add_parser("chunked_hnet")
    chunked_parser.add_argument("env")

    hnet_mt_parser = subparsers.add_parser("hnet_mt")
    hnet_mt_parser.add_argument("env")

    hnet_replay_parser = subparsers.add_parser("hnet_replay")
    hnet_replay_parser.add_argument("env")


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
    # --- possibility of adding ---
    elif args.cmd == "coreset":
        run_coreset(args)
    elif args.cmd == "finetune":
        run_finetune(args)
    elif args.cmd == "single":
        run_single(args)
    elif args.cmd == "si":
        run_si(args)
    elif args.cmd == "multitask":
        run_multitask(args)
    elif args.cmd == "chunked_hnet":
        run_chunked_hnet(args)
    elif args.cmd == "hnet_mt":
        run_hnet_mt(args)
    elif args.cmd == "hnet_replay":
        run_hnet_replay(args)


if __name__ == "__main__":
    main()