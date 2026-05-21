import argparse
import hypercrl
import torch


# ============================================================
# DEVICE PARSING
# ============================================================
def parse_device(device_str: str | None) -> str:
    """
    Central device parsing utility.

    This function normalizes all device inputs into a consistent format:
    - "cpu"
    - "cuda:X"
    - "mps"

    It also applies safe fallbacks if requested devices are unavailable.
    """

    # --------------------------------------------------------
    # DEFAULT: auto-select best available device
    # --------------------------------------------------------
    if device_str is None:
        if torch.cuda.is_available():
            return "cuda:0"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    device_str = device_str.lower()

    # --------------------------------------------------------
    # CPU
    # --------------------------------------------------------
    if device_str == "cpu":
        return "cpu"

    # --------------------------------------------------------
    # CUDA
    # --------------------------------------------------------
    if device_str.startswith("cuda"):
        if not torch.cuda.is_available():
            return "cpu"

        # normalize shorthand "cuda" -> "cuda:0"
        return "cuda:0" if device_str == "cuda" else device_str

    # --------------------------------------------------------
    # Apple Metal Performance Shaders (MPS)
    # --------------------------------------------------------
    if device_str == "mps":
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    # --------------------------------------------------------
    # FALLBACK
    # --------------------------------------------------------
    return "cpu"


# ============================================================
# METHOD WRAPPERS
# ============================================================
"""
Each wrapper function is responsible for:
- forwarding CLI arguments to hypercrl
- ensuring consistent parameter passing
- isolating CLI layer from training logic
"""


def run_hnet(args):
    """Run Hypernetwork-based continual learning."""
    return hypercrl.hnet(
        args.env,
        seed=args.seed,
        savepath=args.savepath,
        play=args.play,
        render=args.rendering,
        device=args.device
    )


def run_pnn(args):
    """Run Progressive Neural Networks baseline."""
    return hypercrl.pnn(
        args.env,
        seed=args.seed,
        savepath=args.savepath,
        render=args.rendering,
        device=args.device
    )


def run_ewc(args):
    """Run Elastic Weight Consolidation (EWC)."""
    return hypercrl.ewc(
        args.env,
        seed=args.seed,
        savepath=args.savepath,
        render=args.rendering,
        device=args.device
    )


def run_coreset(args):
    """Run coreset-based continual learning."""
    return hypercrl.coreset(
        args.env,
        seed=args.seed,
        savepath=args.savepath,
        render=args.rendering,
        device=args.device
    )


def run_finetune(args):
    """Run naive fine-tuning baseline."""
    return hypercrl.finetune(
        args.env,
        seed=args.seed,
        savepath=args.savepath,
        render=args.rendering,
        device=args.device
    )


def run_si(args):
    """Run Synaptic Intelligence (SI)."""
    return hypercrl.si(
        args.env,
        seed=args.seed,
        savepath=args.savepath,
        render=args.rendering,
        device=args.device
    )


def run_multitask(args):
    """Run multitask learning baseline."""
    return hypercrl.multitask(
        args.env,
        seed=args.seed,
        savepath=args.savepath,
        render=args.rendering,
        device=args.device
    )


def run_single(args):
    """Run single-task training baseline."""
    return hypercrl.single(
        args.env,
        seed=args.seed,
        savepath=args.savepath,
        render=args.rendering,
        device=args.device
    )


def run_chunked_hnet(args):
    """Run chunked hypernetwork variant."""
    return hypercrl.chunked_hnet(
        args.env,
        seed=args.seed,
        savepath=args.savepath,
        render=args.rendering,
        device=args.device
    )


def run_hnet_mt(args):
    """Run multitask hypernetwork training."""
    return hypercrl.hnet_mt(
        args.env,
        seed=args.seed,
        savepath=args.savepath,
        render=args.rendering,
        device=args.device
    )


def run_hnet_replay(args):
    """Run hypernetwork with replay buffer."""
    return hypercrl.hnet_replay(
        args.env,
        seed=args.seed,
        savepath=args.savepath,
        render=args.rendering,
        device=args.device
    )


# ============================================================
# METHOD REGISTRY
# ============================================================
"""
Central dispatch table mapping CLI methods to implementations.
This avoids large if/elif chains and simplifies extension.
"""
METHODS = {
    "hnet": run_hnet,
    "pnn": run_pnn,
    "ewc": run_ewc,
    "coreset": run_coreset,
    "finetune": run_finetune,
    "si": run_si,
    "multitask": run_multitask,
    "single": run_single,
    "chunked_hnet": run_chunked_hnet,
    "hnet_mt": run_hnet_mt,
    "hnet_replay": run_hnet_replay,
}


# ============================================================
# CLI ENTRY POINT
# ============================================================
def main():
    """
    Command-line interface entry point.

    Example usage:
        python main.py run --method si --env cartpole --device cpu
    """

    parser = argparse.ArgumentParser(description="HyperCRL training CLI")

    sub = parser.add_subparsers(dest="cmd", required=True)

    # --------------------------------------------------------
    # unified RUN command
    # --------------------------------------------------------
    run_parser = sub.add_parser("run", help="Run a selected method")
    run_parser.add_argument("--method", required=True, choices=METHODS.keys())
    run_parser.add_argument("--env", required=True)
    run_parser.add_argument("--seed", type=int, default=None)
    run_parser.add_argument("--savepath", type=str, default=None)
    run_parser.add_argument("--play", action="store_true")
    run_parser.add_argument("--device", type=str, default=None)
    run_parser.add_argument("--rendering", action="store_true")

    args = parser.parse_args()

    # --------------------------------------------------------
    # DEVICE NORMALIZATION
    # --------------------------------------------------------
    args.device = parse_device(args.device)
    print(f"[DEBUG] Using device: {args.device}")

    # --------------------------------------------------------
    # DISPATCH EXECUTION
    # --------------------------------------------------------
    fn = METHODS[args.method]
    fn(args)


if __name__ == "__main__":
    main()