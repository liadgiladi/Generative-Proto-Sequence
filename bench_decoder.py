"""Microbenchmark: how much does the minGPT decoder cost vs the old MLP decoder?

The VAE decoder went from a single ``nn.Sequential`` forward pass (one shot, fully
batched) to an autoregressive Transformer that runs one trunk pass per generated token.
At our sizes (n_embd=32, vocab=5, seq=10) neither is remotely FLOP-bound -- both are
dominated by per-operation dispatch and kernel launch overhead -- so the thing that
actually sets wall-clock time is *how many ops get dispatched per call*, which went up by
roughly 50x.

Measurements on a GTX 1080 Ti established that cost is ~100% overhead:

  * batch 1 and batch 256 cost the same to within 1% (256x the arithmetic, no time)
  * n_embd 32 -> 16 is 3.5x fewer params and 1.6% faster
  * cost fits ``3.73 ms + 3.91 ms * n_layer``, so even a 0-layer trunk would cost 7x the
    entire old decoder -- that constant is the 10-iteration Python loop itself

Which means the lever that matters is not model size but eliminating the dispatch, so the
second half of this script benchmarks three ways to do that: CUDA graph capture,
torch.compile's reduce-overhead mode, and simply running the tiny decoder on the CPU.

Run it on a GPU node; it needs no dataset:

    sbatch sbatch/e0_bench_decoder.sbatch

Compare the ``per-call`` columns. The MLP row is the 9-hour-run baseline.
"""

from __future__ import annotations

import argparse
import time
import traceback
from typing import Callable, Optional

import torch
import torch.nn as nn

from generative.models.dense_autoencoder.decoder_utils import ActionGen
from generative.models.dense_autoencoder.dense_var_auto_encoder_vin_1 import DenseVAE

SEQ_LEN = 10
N_WORDS = 5
LATENT = 16


class LegacyMlpDecoder(nn.Module):
    """The pre-Transformer decoder trunk, reproduced from commit 47d6da4^.

    Kept here purely as a timing reference: this is the decoder the 9-hour run used.
    """

    def __init__(self, decoder_input_size: int = LATENT, input_length: int = SEQ_LEN,
                 n_words: int = N_WORDS) -> None:
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(decoder_input_size, decoder_input_size),
            nn.InstanceNorm1d(decoder_input_size),
            nn.LeakyReLU(0.2),
            nn.Linear(decoder_input_size, decoder_input_size * 2),
            nn.InstanceNorm1d(decoder_input_size * 2),
            nn.LeakyReLU(0.2),
            nn.Linear(decoder_input_size * 2, input_length * n_words),
            nn.InstanceNorm1d(input_length * n_words),
        )
        self.n_action_seq_length = input_length
        self.n_words = n_words

    def gen_action_seq(self, z: torch.Tensor) -> torch.Tensor:
        """Mirror the old ActionGen.gen_action_seq one-hot path."""
        logits = self.trunk(z).reshape(-1, self.n_action_seq_length, self.n_words)
        return torch.nn.functional.gumbel_softmax(logits, tau=1.0, hard=True)


def timeit(fn: Callable[[], torch.Tensor], device: torch.device,
           n_warmup: int = 20, n_iter: int = 100) -> float:
    """Return mean seconds per call, with the GPU properly synchronised."""
    for _ in range(n_warmup):
        fn()
    if device.type == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(n_iter):
        fn()
    if device.type == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - start) / n_iter


def make_gpt_decoder(device: torch.device, n_layer: int = 2, n_head: int = 4,
                     n_embd: int = 32, dropout: float = 0.0) -> ActionGen:
    vae = DenseVAE(
        input_length=SEQ_LEN, n_words=N_WORDS, device=device,
        decoder_input_size=LATENT,
        decoder_n_layer=n_layer, decoder_n_head=n_head, decoder_n_embd=n_embd,
        decoder_dropout=dropout,
    ).to(device)
    vae.eval()
    return ActionGen(pretrained_decoder=vae, n_act_seq_len=SEQ_LEN, device=device)


ROW_FMT = "{:<30} {:>8}  {:>9}  {:>11}  {:>13}"


def bench_row(label: str, device: torch.device, n_params: Optional[int],
              fwd1: Callable[[], torch.Tensor],
              fwd_batch: Callable[[], torch.Tensor],
              fwd_bwd: Optional[Callable[[], torch.Tensor]] = None) -> None:
    """Time one decoder variant and print it as a table row.

    A variant that cannot be benchmarked (unsupported on this GPU, compile failure) prints
    the reason instead of a row rather than aborting the whole run.
    """
    try:
        with torch.no_grad():
            t1 = timeit(fwd1, device)
            tb = timeit(fwd_batch, device)
        tbwd = timeit(fwd_bwd, device, n_warmup=10, n_iter=50) if fwd_bwd else None
    except Exception:
        print(f"{label:<30} FAILED -- see traceback below")
        traceback.print_exc()
        return

    print(ROW_FMT.format(
        label,
        f"{n_params:,}" if n_params is not None else "-",
        f"{t1 * 1e3:.3f}",
        f"{tb * 1e3:.3f}",
        f"{tbwd * 1e3:.3f}" if tbwd is not None else "-",
    ))


# --------------------------------------------------------------------------------------- #
# Overhead-elimination candidates
# --------------------------------------------------------------------------------------- #

def make_cuda_graph_runner(gen: ActionGen, z_template: torch.Tensor) -> Callable:
    """Capture the whole 10-step generation into a single replayable CUDA graph.

    Capture records the concrete kernel sequence, so the KV cache's growing ``torch.cat``
    shapes are not a problem -- they are fixed constants at capture time. What matters is
    that the input lives at a fixed address (hence ``static_z``, which each call copies
    into) and that nothing in the loop synchronises with the CPU.

    Replaying the graph issues the entire sequence as one submission, removing every bit of
    Python and dispatch overhead -- which is ~100% of this decoder's cost.
    """
    static_z = z_template.clone()

    def run():
        return gen.gen_action_seq(static_z, get_actions_as_one_hot=True)[0]

    # Warm up on a side stream; required before capture so that lazy init (cuBLAS
    # handles, autotuning, allocator blocks) does not land inside the graph.
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            run()
    torch.cuda.current_stream().wait_stream(side)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        static_out = run()

    def replay(z: torch.Tensor) -> torch.Tensor:
        static_z.copy_(z)
        graph.replay()
        return static_out

    return replay


def make_compiled_runner(gen: ActionGen) -> Callable:
    """torch.compile in reduce-overhead mode, which applies CUDA graphs automatically.

    Unlike the manual capture above this needs no restructuring, but it goes through
    Triton, which may not support this GPU's compute capability.
    """
    return torch.compile(
        lambda z: gen.gen_action_seq(z, get_actions_as_one_hot=True)[0],
        mode="reduce-overhead",
    )


def make_cpu_runner(gen_cpu: ActionGen, out_device: torch.device) -> Callable:
    """Run the decoder on the CPU, paying two host transfers per call.

    With no arithmetic to speak of, the CPU has no dispatch-to-device overhead at all. The
    transfers are the whole cost, and this measures them honestly: the real training loop
    gets ``z`` from the GPU actor and feeds the result to the GPU critic.
    """

    def run(z: torch.Tensor) -> torch.Tensor:
        out = gen_cpu.gen_action_seq(z.to("cpu", non_blocking=False),
                                     get_actions_as_one_hot=True)[0]
        return out.to(out_device)

    return run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch", type=int, default=256,
                        help="batch size of the training-loop decoder calls")
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"device: {device}")
    if device.type == "cuda":
        cap = torch.cuda.get_device_capability()
        print(f"gpu: {torch.cuda.get_device_name(0)}  (compute capability sm_{cap[0]}{cap[1]})")
    print(f"torch: {torch.__version__}\n")

    z1 = torch.randn(1, LATENT, device=device)
    zb = torch.randn(args.batch, LATENT, device=device)

    header = ROW_FMT.format("decoder", "params", "fwd b=1",
                            f"fwd b={args.batch}", f"fwd+bwd b={args.batch}")
    print(header)
    print("-" * len(header))
    print("(all times are ms per call)")

    # --- Reference: the decoder the 9-hour run used -----------------------------------
    mlp = LegacyMlpDecoder().to(device)
    bench_row(
        "MLP (legacy, 1 pass)", device,
        sum(p.numel() for p in mlp.parameters()),
        lambda: mlp.gen_action_seq(z1),
        lambda: mlp.gen_action_seq(zb),
        lambda: mlp.gen_action_seq(zb).sum().backward(),
    )

    # --- The Transformer decoder, eager, at a few trunk sizes -------------------------
    for n_layer, n_head, n_embd in [(2, 4, 32), (1, 4, 32), (1, 2, 16)]:
        gen = make_gpt_decoder(device, n_layer, n_head, n_embd)
        bench_row(
            f"GPT eager L={n_layer} embd={n_embd}", device,
            sum(p.numel() for p in gen.model.decoder.parameters()),
            lambda g=gen: g.gen_action_seq(z1, get_actions_as_one_hot=True)[0],
            lambda g=gen: g.gen_action_seq(zb, get_actions_as_one_hot=True)[0],
            lambda g=gen: g.gen_action_seq(zb, get_actions_as_one_hot=True)[0].sum().backward(),
        )

    # --- Overhead elimination, all at the production shape (L=2, embd=32) -------------
    print()
    print("-- overhead elimination (same L=2 embd=32 decoder, forward only) --")

    gen = make_gpt_decoder(device, 2, 4, 32)
    n_params = sum(p.numel() for p in gen.model.decoder.parameters())

    if device.type == "cuda":
        # A captured graph is bound to one input shape, so batch 1 and batch N need
        # separate captures -- which is fine, the training loop only uses a couple of
        # fixed shapes.
        try:
            with torch.no_grad():
                replay1 = make_cuda_graph_runner(gen, z1)
                replayb = make_cuda_graph_runner(gen, zb)
            bench_row("GPT cuda-graph", device, n_params,
                      lambda: replay1(z1), lambda: replayb(zb))
        except Exception:
            print("GPT cuda-graph                 FAILED -- see traceback below")
            traceback.print_exc()

        try:
            compiled = make_compiled_runner(gen)
            # Compilation happens on first call and can take a minute; it is inside the
            # warmup, so it does not pollute the timings.
            bench_row("GPT torch.compile(r-o)", device, n_params,
                      lambda: compiled(z1), lambda: compiled(zb))
        except Exception:
            print("GPT torch.compile(r-o)         FAILED -- see traceback below")
            traceback.print_exc()

    gen_cpu = make_gpt_decoder(torch.device("cpu"), 2, 4, 32)
    cpu_run = make_cpu_runner(gen_cpu, device)
    bench_row("GPT on cpu (+transfers)", device, n_params,
              lambda: cpu_run(z1), lambda: cpu_run(zb))

    print()
    print("How to read this: the decoder's share of a 1,000,000-step run is roughly")
    print("  training loop = 1,000,000 * (fwd b=256)        [target-Q, every step]")
    print("                + 250,000 * (fwd+bwd b=256)      [actor update, every 4th step]")
    print("                + ~167,000 * (fwd b=1)           [action selection]")
    print("  evaluation    = n_evals * n_episodes * gens_per_episode * (fwd b=1)")
    print("gens_per_episode is ~8 with healthy sequences and saturates at max_episode_steps")
    print("(75) when the decoder degenerates to length-1 sequences, so it swings the eval")
    print("term by ~10x. Watch 'mean num valid actions in sequences' in the training log.")


if __name__ == "__main__":
    main()
