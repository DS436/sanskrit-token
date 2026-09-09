# Renting a GPU to run the Experiment 05 sweep

This is a procedure for someone who has never rented a cloud GPU. It assumes you can use a
terminal and `ssh`, and nothing else. It assumes no knowledge of this project: everything
the sweep needs is stated here, with the file each number came from.

The companion document is [`README.md`](README.md) in this directory, which explains what
the experiment measures and why. This file only covers getting it to run on hardware you
pay for by the hour. Where the two disagree about a command, the README wins; this file
adds the steps around it.

Prices in section 3 were read off the providers' own pricing pages on the dates given. GPU
rental prices move. Re-read the pages before you spend anything.

---

## 1. What the job is, in numbers

Read these before choosing hardware. Every one of them comes from a file in this
repository, named in the last column.

| quantity | value | source |
|---|---|---|
| runs in the sweep | **51** (Track 1: 21, Track 2: 30) | `sweep.yaml` grid; `README.md`, "Projected cost of the real sweep" totals |
| what a run is | one arm, one model size, one seed, trained from scratch | `sweep.yaml`: `tracks`, `sizes`, `seeds: [0, 1, 2]` |
| total training tokens | **11,734,624,761** | `README.md`, projection table, GRAND TOTAL row |
| optimizer steps, whole sweep | **~179,000** (11,734,624,761 ÷ 65,536) | tokens above ÷ `README.md`, "VRAM and batch size": 65,536 tokens per step |
| training compute | **5,119.5 PFLOPs** (Track 1 486.9, Track 2 4,632.6) | `README.md`, projection table |
| evaluation compute | **941 PFLOPs**, a further 18% | `README.md`, "Evaluation is not free"; `docs/decisions.md`, 2026-09-06, "MEASURED: the phase-A fix wave" |
| total compute | **6,061 PFLOPs = 6.06 × 10¹⁸ FLOPs** | the two rows above; `README.md` cost table |
| A100 floor | **~13.5 A100-hours** at 40% model-FLOPs-utilisation | `README.md` cost table |
| A100 realistic | **18 to 26 A100-hours** at the 20-30% MFU this model actually reaches | `README.md`, "The realistic budget"; `docs/decisions.md`, 2026-09-06, "Sweep hardening before paid GPU time", item (5) |
| H100 floor | ~4.3 hours at 40% MFU | `README.md` cost table |
| upload: corpora and held-out texts | **837 MB** (`data/processed/lm/` without `track2_raw.txt`) | `README.md` rsync table rows: 32 + 34 + 764 + 7.0 MB; confirmed by `du` on the machine that built them |
| upload: tokenizers | **26 MB** (six arms) | `README.md` rsync table |
| deliberately not uploaded | `track2_raw.txt`, 2.4 GB | `README.md`: "the sweep reads `track2_sample.txt`" |
| smoke sweep | 3 arms, 1 seed, 300 steps; **89 s of training** on a laptop GPU, about 5 minutes cold | `README.md`, "Smoke sweep" and step 3 of the GPU procedure |
| Python environment | **3.0 GB of wheels** to download on Linux (torch 2.14.0 plus the NVIDIA CUDA 13 runtime packages) | summed from `uv.lock` |
| working disk | ~837 MB corpora, ~3.5 GB encoded token cache, ~1.0-1.6 GB peak checkpoint; **budget 8 GB beyond the environment** | `README.md`, "Disk" |
| results to bring home | four small files per run (20 KB measured on a smoke run) plus four figures; **tens of MB** | measured on `outputs/05_lm_training/smoke/` |

One number in the README's rsync table is wrong and worth knowing about: the subtotal row
says 809 MB, but the rows above it sum to 837 MB, which is what `du` reports. Use 837 MB.

Two consequences of the table. First, the honest planning figure is the FLOPs, not the
hours: 6,061 PFLOPs is a physical quantity, and hours are that divided by a throughput you
have not measured yet. Second, the sweep is one long sequential job, not a burst, so what
you are buying is a machine that stays up for most of a day.

---

## 2. What to rent

### One GPU, not several

The sweep trains **one run at a time in one process on one device**. `run_sweep` in
`sweep.py` iterates over pending runs in a plain Python `for` loop and calls `run_one` on
each; there is no `torch.distributed`, no `DistributedDataParallel`, and no `torchrun`
entry point anywhere in `src/` or `experiments/`. `resolve_device` in
`src/sanskrit_tok/lm/config.py` returns the single string `"cuda"`, which means device 0.

**Renting two GPUs will not halve the time.** A second card sits idle. Worse, starting a
second copy of the same command against the same `output_dir` does not split the work:
`pending_runs` is computed once at startup from which `results.json` files exist, so both
processes would take the same list in the same order and train the same run twice.

There is a manual way to use two cards, if you want it. Split the grid into two config
files, one per track (Track 1 is 21 runs and about 11% of the FLOPs, Track 2 is 30 runs and
89%, so this is a poor split), or split Track 2 by size, give each config its own
`output_dir`, and run them under separate `CUDA_VISIBLE_DEVICES`. You then have to
aggregate across two output directories by hand. For a job this size that is not worth the
risk of getting the plan hashes or the reference arm wrong. **Rent one card.**

### How much VRAM

The memory that decides this is not the model. It is the logits tensor, and the README
works it out under "A note on the VRAM numbers":

- at the 50M size with `batch_size: 16`, the logits are `16 × 1024 × 64,064 × 4 bytes`
  ≈ **4.2 GB**, materialised in fp32 even under bf16 autocast, because
  `masked_cross_entropy` writes `-inf` into the padding columns and `F.cross_entropy`
  promotes. With its gradient and the bf16 copy, peak is **~14 GB**.
- at the 125M size with `batch_size: 8`, ~2.1 GB of logits and **~9-11 GB** peak.

So **24 GB is enough** with the settings as shipped in `sweep.yaml`, and the README's card
table starts there for that reason. 40 GB and 80 GB cards let you raise `batch_size` and
lower `grad_accum`, which is slightly faster but changes nothing about the results as long
as the product `batch_size × grad_accum × block_size` stays at **65,536 tokens per step**.
Change the split, never the product; the byte budget is spent in units of that product and
runs with a different one are not comparable to the reported ones.

### Ampere or newer, which is not negotiable

`resolve_dtype` in `src/sanskrit_tok/lm/config.py` selects bf16 only where
`torch.cuda.is_bf16_supported()` is true, which means compute capability 8.0 or above.
On an older card it silently falls back to fp16 with a `GradScaler`, a path this project
has never validated on this model, and which can turn a NaN'd loss into a missing number in
`results.json` rather than an error. **Do not rent a V100 or a T4.** Ampere or newer means
A100, A10, A10G, A6000, L4, L40S, RTX 4090, H100, and their successors.

### Would a smaller or older card do?

Yes, at the cost of wall clock, as long as it is Ampere or newer and has at least 24 GB.
The job is 6,061 PFLOPs whatever you run it on, so the hours scale roughly inversely with
the card's usable bf16 throughput. The README's own conversion is the model: A100 at 40%
MFU is 13.5 hours and is a floor; 20-30% MFU is 18 to 26 hours and is the planning figure.
A card with half an A100's bf16 tensor throughput will take roughly twice as long, and if
it is cheaper than half the price of an A100 it is the better buy. Section 3 gives the
arithmetic per card.

Two caveats on the small-card route. Memory bandwidth matters more than peak FLOPs for a
model this small with a 64k-wide softmax, so the scaling is approximate. And a longer job
means more hours of exposure to interruption, which section 5 covers.

---

## 3. Where to rent, and what it costs

Every price below was read off the provider's own page on **2026-09-09** and the URL is
given. Prices exclude tax. They change; check before you spend.

**The short answer**, if you do not want to read the rest: rent **one A100 on RunPod**, about
$1.19 to $1.59 an hour, expect **18 to 26 hours** and a bill around **$30**, and use
on-demand rather than spot. If you would rather pay less and wait longer, rent **one RTX
A6000 at about $0.33 an hour**, expect 36 to 52 hours and a bill around **$15**. Everything
below is the working for those two sentences.

### The providers, compared

These all rent a single GPU by the hour to an individual with a card, with no enterprise
contract, and give you a machine you can `ssh` into.

| provider | relevant cards, $/GPU/hour | storage | egress | minimum | payment | how fast you get it |
|---|---|---|---|---|---|---|
| **RunPod** (`runpod.io/pricing`) | Community: RTX A6000 **0.33**, RTX 4090 **0.34**, L4 0.44, RTX 5090 0.69, L40S 0.79, A100 PCIe 80GB **1.19**, A100 SXM 1.39, H100 PCIe 1.99. Secure: A6000 0.53, 4090 0.74, L40S 1.09, A100 1.59, H100 PCIe 2.89 | extra: container and running volume disk $0.10/GB/mo, idle volume $0.20/GB/mo, network volume $0.07/GB/mo under 1 TB | "no fees for data transfer" | must hold one hour's credit; credits from $10, but **$100 minimum per transaction for prepaid cards** | Stripe card, crypto (KYC), invoicing over $5,000 | prepaid credit, deploy is immediate subject to stock |
| **Lambda** (`lambda.ai/service/gpu-cloud`, `docs.lambda.ai/public-cloud/billing/`) | 1x: A6000 **1.09**, A10 1.29, A100 40GB (SXM or PCIe) **1.99**, GH200 2.29, H100 PCIe 3.29, H100 SXM 4.29. A100 80GB appears only in the 8x SXM config at 2.79 | extra: $0.20/GiB/month, billed in one-hour increments, no minimum period | "you are not charged for ingress or egress" | none for on-demand; billed per minute | not stated on the pages read | "first-come access"; capacity behaviour not stated on the pages read |
| **Vast.ai** (`vast.ai/pricing`; **the page prints no prices**, figures below are live single-GPU on-demand listings from the public offers API, median / cheapest) | RTX 4090 **0.455 / 0.136**, RTX 5090 0.468 / 0.282, RTX A6000 **0.329 / 0.228**, A100 SXM4 0.835 / 0.336, L40S 0.801 / 0.469, H100 SXM 1.869 / 1.336 | extra: set by the host, median ~$0.20/TB/month, **charged while the instance is stopped as well as running** | **charged**, median ~$4.00/TB each way | prepaid credits | prepaid credits | immediate, but it is a marketplace: the cheapest listings are unverified hosts in arbitrary countries. Filter on the reliability score |
| **Paperspace** (`paperspace.com/pricing`, now serving DigitalOcean-branded content) | A100 **1.94**, A6000 1.89, A5000 1.38, RTX5000 0.82, A4000 0.76, H100 5.95. A100-80G at 1.15 is printed **only** against a three-year commitment | overage $0.29/GB | not stated on the page | Gradient tiers $0/$8/$39 per month, billed separately from instance hours | not stated on the page | not stated on the page |
| **Hyperstack** (`hyperstack.cloud/gpu-pricing`) | **A6000 0.50**, A4000 0.15, L40 1.00, A100 **1.35**, A100 NVLink 1.40, A100 SXM 1.60, H100 2.50 | extra: $0.000096774/GB/hour (about $0.07/GB/month); public IP $0.00672043/hour | "Free" both ways | none stated; prepaid accounts, billed per minute | prepaid | not stated on the page |
| **Crusoe** (`crusoe.ai/cloud/pricing/`) | L40S 1.50, A100 80GB PCIe **2.00**, A100 80GB SXM 2.30, H100 80GB HGX 3.90 | extra: persistent disk $0.08/GiB/month | "does not charge for network ingress or egress" | "no upfront minimums or setup fees for on-demand" | not stated on the page | not stated on the page |
| **Verda**, formerly DataCrunch (`verda.com/pricing`) | on-demand / spot: A100 40GB **1.29 / 0.645**, A100 80GB **1.79 / 0.895**, RTX 6000 Ada 1.04 / 0.52, L40S 1.37 / 0.685, H100 SXM5 3.25 / 1.63 | extra: $0.20/GiB/month | none stated | none stated | not stated on the page | not stated on the page |

Two providers that look relevant and are not:

- **Modal** (`modal.com/pricing`) prices A100 40GB at $0.000583/second, which is $2.10/hour,
  A100 80GB at $2.50/hour, L4 at $0.80/hour, H100 SXM5 at $3.95/hour, plus CPU at
  $0.0000131 per core-second and memory at $0.00000222 per GiB-second on top, with $30/month
  of free credit on the Starter plan and volumes at $0.09/GiB/month including 1 TiB free.
  But Modal is a serverless function platform, not a machine you `ssh` into. Running this
  sweep there means rewriting `run.py` as a Modal app with a Volume for the corpora and the
  outputs. That is a day of work to save a few dollars. **Not recommended for this job.**
- **Together AI** (`together.ai/pricing`) sells GPU clusters: H100 at $3.99/GPU/hour
  on-demand, $1.99/hour preemptible, $3.19 to $3.69 reserved. There is no self-serve
  single-GPU hourly flow on the page, and storage, egress and minimum term are not
  disclosed. Treat it as a sales conversation, not a rental.

For reference, the hyperscalers, which are the expensive end (all read 2026-09-09):

- **AWS**, US East (N. Virginia), Linux, from the official on-demand price feed:
  `g5.xlarge` (1x A10G) $1.0060/hour, `g6.xlarge` (1x L4) $0.8048/hour, `g6e.xlarge`
  (1x L40S) $1.8610/hour, `p4d.24xlarge` (8x A100 40GB) $21.9576/hour, which is
  $2.74/GPU-hour.
- **GCP**, Iowa: `a2-highgpu-1g` (1x A100 40GB) **$3.673385/hour** on-demand, **$2.12081/hour**
  spot; `a2-ultragpu-1g` (1x A100 80GB) $5.06879789/hour. A100, H100 and L4 are not sold as
  standalone GPU attachments, only inside A2/A3/G2 machine types.
- **Azure**, East US, from the retail prices API: `Standard_NC24ads_A100_v4` (1x A100 80GB),
  Linux, **$3.673/hour** on-demand and **$0.67877/hour spot**. That spot price is the
  cheapest A100 anywhere in this section, but it is the only thing about Azure that is
  cheap here.

### Turning hours into money

The repository's figure is **18 to 26 A100-hours** (`README.md`, "The realistic budget").
Applied to the A100 prices above:

| where | $/hour | 18 h | 26 h |
|---|---:|---:|---:|
| Vast.ai A100 SXM4, median listing | 0.835 | $15 | $22 |
| RunPod Community, A100 PCIe 80GB | 1.19 | $21 | $31 |
| Hyperstack A100 | 1.35 | $24 | $35 |
| RunPod Secure, A100 | 1.59 | $29 | $41 |
| Verda A100 80GB on-demand | 1.79 | $32 | $47 |
| Paperspace A100 | 1.94 | $35 | $50 |
| Lambda A100 40GB | 1.99 | $36 | $52 |
| Crusoe A100 80GB PCIe | 2.00 | $36 | $52 |
| Modal A100 80GB (not recommended, see above) | 2.50 | $45 | $65 |
| GCP `a2-highgpu-1g` on-demand | 3.673 | $66 | $95 |

Storage and transfer are noise at this scale. A 60 GB volume for a day is about $0.20 on
RunPod's running-volume rate and about $0.40 on Lambda's. The upload is 863 MB and the
download is tens of MB, so even Vast's ~$4/TB is under a cent. **The hourly rate is the
whole cost.** Do not optimise anything else.

### Cheaper cards, longer job

The job is a fixed 6,061 PFLOPs, so hours scale roughly with the card's bf16 throughput.
The relevant number is bf16 tensor throughput **with FP32 accumulate**, dense, because that
is what PyTorch's bf16 autocast does. NVIDIA publishes it, and for consumer cards it is
half the headline figure. All specifications read 2026-09-09:

| card | VRAM | CC | dense bf16 with FP32 accumulate | vs A100 | hours (from 18-26 A100-h) | cheapest listed $/h | total |
|---|---:|---:|---:|---:|---:|---:|---:|
| A100 80/40 GB | 80 / 40 | 8.0 | **312 TFLOP/s** | 1.00 | 18-26 | 1.19 (RunPod Community) | $21-31 |
| RTX A6000 | 48 | 8.6 | **154.8** | 0.50 | 36-52 | 0.33 (RunPod Community) | **$12-17** |
| RTX 4090 | 24 | 8.9 | **165.2** | 0.53 | 34-49 | 0.34 (RunPod Community) | **$12-17** |
| RTX 5090 | 32 | 12.0 | **209.5** | 0.67 | 27-39 | 0.69 (RunPod Community) | $19-27 |
| A10 | 24 | 8.6 | **125** | 0.40 | 45-65 | 1.29 (Lambda) | $58-84 |
| H100 PCIe | 80 | 9.0 | **756.5** | 2.42 | see below | 1.99 (RunPod Community) | $18-26 |
| H100 SXM | 80 | 9.0 | **989.5** | 3.17 | see below | 2.69 (RunPod Community) | $24-35 |

Sources for the throughput column: NVIDIA A100 datasheet, which prints
`BFLOAT16 Tensor Core 312 TFLOPS | 624 TFLOPS*` with `* With sparsity`, so 312 is already
the dense figure; the NVIDIA Ampere GA102 whitepaper Table 3 for the A6000
(`Peak BF16 Tensor TFLOPS with FP32 Accumulate 154.8/309.6`); the NVIDIA Ada whitepaper
Appendix A for the RTX 4090 (`Peak BF16 Tensor TFLOPS with FP32 Accumulate 165.2/330.4`,
exactly half its FP16-accumulate rate); the NVIDIA RTX Blackwell whitepaper for the 5090
(`209.5/419`); the NVIDIA A10 datasheet (`BFLOAT16 Tensor Core 125 TF | 250 TF*`); and the
NVIDIA H100 datasheet, whose figures are sparsity-doubled and are halved here (1,979 → 989.5
SXM, 1,513 → 756.5 PCIe). Compute capabilities from `developer.nvidia.com/cuda-gpus`.

Read the scaling in one direction only. **Downwards it is reliable**: a card at half an
A100's throughput running the same kernels takes about twice as long, and the 20-30% MFU
band the README uses already absorbs the difference in memory bandwidth. **Upwards it is
not**: a 50M-parameter model at batch 16 will not keep an H100 fed anywhere near 40% MFU,
which is why the README says "budget half" the A100 figure on an H100, meaning **9 to 13
hours**, not the 4.3-hour FLOP floor. The H100 rows above use that, not the ratio.

Two cards deliberately left out. **L4 and L40S**: NVIDIA does not state whether their
published bf16 figures (121 and 362.05 TFLOP/s dense) are FP32-accumulate or
FP16-accumulate, and the Ada whitepaper quotes the near-identical L40 at half the L40S
number, so the honest range for an L40S is 181 to 362 and for an L4 is 60.5 to 121. Sizing
against a number that might be double the truth is how a rental overruns. If you want one,
benchmark it with the smoke sweep first.

**The cheapest sensible choices are an A6000 or a 4090 at about $0.33/hour**, roughly $12 to
$17 for the whole sweep over a day and a half to two days. The A6000 is the safer of the two
because 48 GB lets you keep the shipped `batch_size: 16`; on a 24 GB card the README's table
says to use `batch_size: 8, grad_accum: 8` at 50M and `batch_size: 4, grad_accum: 16` at
125M, keeping the 65,536-token product.

The trade against the A100 is exposure. A two-day job has two days in which to be
interrupted, and it is two days of watching. Given the totals involved, paying about $30 to
have it finished inside one day on an A100 is a reasonable thing to do.

Do not go below 24 GB and do not go below Ampere, for the reasons in section 2.

### Spot and interruptible, and whether this job can use them

**It can, with one condition.** The sweep checkpoints every 25 steps at the 50M size and
every 100 at 125M, resumes a part-finished run from `ckpt.pt`, and skips already-finished
runs by plan hash. That is established from the code, not assumed; the evidence is in
section 5. **But resumption reads files from disk.** A reclaim that keeps the volume costs
you minutes. A reclaim that destroys the disk costs you everything not already copied off.

So: use spot only where the working directory is on a **persistent volume that survives the
reclaim**, and copy the results directory back to your laptop every few hours regardless.

What is actually on offer, read 2026-09-09:

- **Vast.ai interruptible**, described on its pricing page as "50%+ cheaper" and
  "Preemptible, may be reclaimed"; you bid, and a higher bidder preempts you. Live median
  bid prices: RTX A6000 $0.233, RTX 4090 $0.347, A100 SXM4 $0.667, H100 SXM $1.150.
- **Verda** spot is a flat 50% of on-demand: A100 80GB $0.895/hour, A100 40GB $0.645/hour.
- **Azure** spot `Standard_NC24ads_A100_v4` (1x A100 80GB) at **$0.67877/hour**.
- **GCP** spot `a2-highgpu-1g` (1x A100 40GB) at $2.12081/hour.
- **RunPod publishes no spot discount at present.** The pricing page shows none, the pod
  pricing docs describe only on-demand and 3/6-month savings plans, and RunPod's own public
  API returns a minimum bid price equal to the on-demand price for every GPU. Older RunPod
  blog copy quotes a roughly 50% spot discount; that is a blog example, not a current rate,
  and should not be planned against.

Given that the on-demand total is $12 to $50, spot saves you tens of dollars and buys you a
class of failure you have to manage. **For a first rental, use on-demand.**

### The total, including everything that is not the sweep

The sweep is not the only thing you pay for. The billing clock starts when the instance
boots, not when training starts, and it keeps running through the setup, the checks, and any
attempt you abandon.

| item | billed time | note |
|---|---|---|
| `uv sync` | 5-10 min | 3.0 GB of wheels (`uv.lock`) |
| `rsync` of 863 MB | 2-25 min | governed by **your upload speed**, not the box's. 863 MB is 12 minutes at 10 Mbit/s and 2 minutes at 50 Mbit/s |
| smoke sweep | ~5 min | `README.md`, step 3 of the GPU procedure |
| dry run | a few min | it samples the corpora to get bytes-per-token |
| first Track 2 encode | ~15 min | measured on a laptop at 154-1,718 MB/min per arm across 764 MB × 5 arms; skipped if you copied the encode cache |
| **the sweep itself** | 18-26 h on an A100, 36-52 h on an A6000 or 4090 | `README.md`, "The realistic budget" |
| one abandoned attempt | 1 h | the realistic failure is an image whose driver is too old, and steps 5 and 7 catch it in the first ten minutes |
| storage | for the whole life of the instance | 60 GB at RunPod's $0.10/GB/month running rate is about $0.20/day; at Lambda's $0.20/GiB/month, about $0.40/day |
| transfer | 863 MB up, tens of MB down | free at RunPod, Lambda, Hyperstack and Crusoe; about $0.004 at Vast's ~$4/TB |

**Cheapest plausible: about $15.** An A6000 or 4090 at $0.33 to $0.34/hour on RunPod
Community, about 38 hours including setup and the smoke test, plus about $0.50 of storage
and no transfer charge. This assumes nothing goes wrong twice and that you are content to
watch a job for a day and a half.

**Comfortable: about $50.** An A100 at $1.19 to $1.59/hour on RunPod Community or Secure,
26 hours of sweep plus 2 hours of setup, checks and one abandoned attempt, storage included.
On Lambda at $1.99/hour the same shape is about $62.

**Load more onto the account than you plan to spend.** RunPod requires at least one
hour's credit to deploy and has a $100 minimum per transaction for prepaid cards, so the
amount you load is not necessarily the amount you spend. Budget **$75** and expect change.

The one way this becomes expensive is leaving the instance running after the sweep finished.
The sweep is 18 to 52 hours; a week of an idle A100 is $200. Set a calendar reminder, and do
step 13.

---

## 4. The procedure

Commands are given in full. Placeholders are written in angle brackets:
`<user>`, `<host>`, `<port>`. Everything else is literal.

### Step 1. Create an account and add payment

Sign up with the provider you chose. Add a card or buy prepaid credit, whichever the
provider uses (section 3 says which for each). Expect to put a small amount on first, not
your whole budget: you want the first hour to be cheap if something is wrong with the
image.

Add your SSH public key in the provider's dashboard **before** creating the instance. Most
providers bake the key into the machine at boot and cannot add one afterwards. If you do
not have a key:

```bash
ssh-keygen -t ed25519 -C "exp05"        # accept the default path, set a passphrase
cat ~/.ssh/id_ed25519.pub               # paste this into the provider's SSH key page
```

### Step 2. Pick and start the instance

Choose:

- **GPU**: one card, Ampere or newer, 24 GB or more (section 2).
- **Disk**: at least **60 GB**. The README says to budget 8 GB of working data beyond the
  environment, and the environment itself is 3.0 GB of wheels before unpacking. Provider
  default container disks of 20 GB are too small. Disk is cheap; running out mid-sweep is
  not.
- **Image**: any recent PyTorch or CUDA image. You do not need their PyTorch, since `uv`
  installs its own, but their image sets the NVIDIA driver, and the driver is the thing
  that can be too old. See step 5.
- **Region**: whichever is closest to you, because you are uploading 863 MB.

Start it. Note the SSH command the dashboard gives you.

### Step 3. Connect

```bash
ssh -i ~/.ssh/id_ed25519 -p <port> <user>@<host>
```

If the provider gives you a full command, use theirs. Everything from here runs on the
rented box unless the prompt says otherwise.

### Step 4. Get the code and the environment

```bash
# uv, the package manager this project uses. Nothing else is needed.
curl -LsSf https://astral.sh/uv/install.sh | sh
source "$HOME/.local/bin/env"

git clone https://github.com/DS436/sanskrit-token.git
cd sanskrit-token
uv sync
```

`uv sync` reads `uv.lock` and installs exactly what it pins: torch 2.14.0, and on Linux the
CUDA 13 runtime packages (`nvidia-cudnn-cu13`, `nvidia-nccl-cu13`, `cuda-toolkit`, `triton`
and the rest). That is 3.0 GB of downloads and takes a few minutes. You do **not** need to
install CUDA yourself and you do **not** need a special PyTorch index: the CUDA build is
what the lock resolves on Linux x86_64.

### Step 5. Check the GPU before you go further

```bash
nvidia-smi
```

Read two things off it: the GPU name, and the **driver version**. The torch build this
project pins is a CUDA 13 build, and CUDA 13.x requires an NVIDIA Linux driver of **580 or
newer** (NVIDIA CUDA Toolkit Release Notes, Table 3, "CUDA minor version compatibility",
read 2026-09-09:
`https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html`). If `nvidia-smi`
reports an older driver, destroy the instance and pick a newer image rather than fighting
it.

Then ask torch directly, which is step 2 of the README's procedure:

```bash
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available(), \
    torch.cuda.get_device_name(0), torch.cuda.is_bf16_supported())"
```

You want four things: a `2.14.0`, a `True`, the card you paid for, and a second `True`.
If the last one is `False` the card is pre-Ampere and the sweep will run an unvalidated
fp16 path; stop and change cards.

### Step 6. Upload the data

The corpora and the trained tokenizers are gitignored, because they are large and derived,
so `git clone` gives you code and nothing to train on. Do step 4 first, because these copy
into the cloned repository. Then run these **from the machine that already has them** (your
laptop, in the repository root), not from the rented box:

```bash
rsync -av --exclude 'track2_raw.txt' \
    data/processed/lm/ <user>@<host>:sanskrit-token/data/processed/lm/

rsync -av \
    outputs/tokenizers/T1_bpe_raw_64k_dcs \
    outputs/tokenizers/T2_unigram_raw_64k_dcs \
    outputs/tokenizers/T4_bpe_split_64k_oracle_dcs \
    outputs/tokenizers/T5_morphbpe_rawseg_64k_dcs \
    outputs/tokenizers/T5_morphbpe_raw_64k_dcs \
    outputs/tokenizers/T6_morphbpe_split_64k_dcs \
    <user>@<host>:sanskrit-token/outputs/tokenizers/
```

Add `-e 'ssh -p <port> -i ~/.ssh/id_ed25519'` to both if the provider uses a non-standard
port. `rsync` resumes: if the transfer dies, run the identical command again.

What that moves, per the README's rsync table:

| what | size | needed for |
|---|---:|---|
| `data/processed/lm/track1_raw.txt` | 32 MB | Track 1, raw arms |
| `data/processed/lm/track1_split.txt` | 34 MB | Track 1, split arms |
| `data/processed/lm/track2_sample.txt` | 764 MB | Track 2, all arms |
| `data/processed/lm/heldout_*.txt` (11 files) | 7.0 MB | every evaluation |
| manifests (`*.manifest.json`, `manifest.json`) | 30 KB | corpus byte counts |
| the six file-backed tokenizer arms | 26 MB | every arm except `T7_byt5` |
| **total** | **863 MB** | |

`T7_byt5` needs no file: its ids are the 256 UTF-8 byte values, constructed in
`src/sanskrit_tok/tokenizers/registry.py` (`ByteAdapter`), so nothing is downloaded for it.
Once step 6 is done the box needs no network access at all until you copy the results back.

`data/processed/lm/track2_raw.txt` is 2.4 GB and is the *unsampled* corpus. The
`--exclude` above is deliberate: the sweep reads `track2_sample.txt`.

**Optionally** also copy the encoded token cache, which saves the box from re-tokenising:

```bash
ssh <user>@<host> 'mkdir -p ~/sanskrit-token/outputs/05_lm_training/encoded'
rsync -av outputs/05_lm_training/encoded/ \
    <user>@<host>:sanskrit-token/outputs/05_lm_training/encoded/
```

This is safe. `ensure_encoded_corpus` in `src/sanskrit_tok/lm/data.py` validates every
cached `.bin` against the corpus's path, byte count and sha256, and against the tokenizer
file's own sha256, and re-encodes if any of them disagrees. A stale cache cannot silently
train a model on the wrong tokens. It is also not worth much: encoding measured on a laptop
runs at 154 MB/min (`T2_unigram_raw_64k_dcs`), 248 MB/min (`T1_bpe_raw_64k_dcs`) and
1,718 MB/min (`T7_byt5`), so the five Track 2 arms encode the 764 MB sample in roughly 15
minutes in total, once, on first use. Skip the copy if it is awkward.

### Step 7. Run the smoke sweep, which is not a result

```bash
cd ~/sanskrit-token
uv run python experiments/05_lm_training/run.py --sweep smoke
```

Three arms, one seed, 300 steps of a two-layer 128-wide model. About five minutes,
including the first encoding of `track1_raw.txt`. It exists to prove the whole pipeline
runs on this box before you commit a day of billing to it, and it says nothing whatever
about tokenizers.

Expected output, in the shape of the laptop run recorded in
`outputs/05_lm_training/smoke/rerun.log`:

```
... INFO exp05.sweep | device cuda, dtype bfloat16 (allow_cpu=True)
... INFO exp05.sweep | 3 runs planned, 3 pending
... INFO exp05.sweep | [1/3] track1/smoke/T1_bpe_raw_64k_dcs/seed0
... INFO sanskrit_tok.lm.train | T1_bpe_raw_64k_dcs: track1_raw.txt -> 6809854 tokens
      (4.631 bytes/token), device cuda, dtype bfloat16
... INFO sanskrit_tok.lm.train | step 300/300 loss 9.1014 bpc {'heldout_dcs': 2.8259, ...}
... INFO exp05.sweep | sweep finished: 3 runs trained this invocation
... INFO exp05.run | done: .../outputs/05_lm_training/smoke
```

The three things to check:

1. The first line says **`device cuda, dtype bfloat16`**. Anything else and you are not
   using the card you are paying for.
2. BPC falls monotonically in every arm. The laptop run went 3.5228 → 2.8744 → 2.8446 →
   2.8259 for `T1_bpe_raw_64k_dcs` and 8.2361 → 3.9841 → 3.8198 → 3.7855 for `T7_byt5`.
3. It ends with `sweep finished` and `done:`, and writes
   `outputs/05_lm_training/smoke/results.json` plus `bpc_curves.pdf` and `bpc_final.pdf`.

Your BPC numbers will be close to the laptop's but **not identical**, and that is expected:
the laptop ran float32 on Apple MPS, the rented box runs bf16 on CUDA. Agreement to two
decimal places is fine. A wildly different number, or a `nan`, is not.

The tokens/s figure in the step lines is your first real measurement of this box. On the
laptop it was about 14,500 for the 64k arms (`README.md`, smoke table). Whatever it is, it
comes from a 8.6M-parameter model and does not transfer to the research sizes, but a figure
far below the laptop's means something is wrong.

### Step 8. Look at the plan before paying for it

```bash
uv run python experiments/05_lm_training/run.py --sweep sweep --dry-run \
    --tokens-per-s 100000
```

Trains nothing. It enumerates all 51 runs, measures or samples each arm's bytes-per-token,
converts the byte budget into a token budget per run, and prints the projection table that
appears in `README.md` under "Projected cost of the real sweep". Check that it says
**51 runs** and about **11.7 billion tokens** and about **5,120 PFLOPs** of training. The
two hour columns depend on the `--tokens-per-s` you passed, which is a guess; ignore them
and trust the FLOPs.

### Step 9. Launch the real sweep, detached

`tmux` is the safer of the two options because it lets you reattach and see the terminal
as it was:

```bash
tmux new -s sweep
cd ~/sanskrit-token
mkdir -p outputs/05_lm_training/sweep
uv run python experiments/05_lm_training/run.py --sweep sweep \
    2>&1 | tee outputs/05_lm_training/sweep/run.log
```

Then press `Ctrl-b` then `d` to detach, and `tmux attach -t sweep` to come back.

If `tmux` is not installed, `nohup` does the same job without the reattach:

```bash
cd ~/sanskrit-token
mkdir -p outputs/05_lm_training/sweep
nohup uv run python experiments/05_lm_training/run.py --sweep sweep \
    > outputs/05_lm_training/sweep/run.log 2>&1 &
tail -f outputs/05_lm_training/sweep/run.log
```

Either way, **wait for the first line and read it before you walk away**. It says
`device cuda, dtype bfloat16` if the card is being used, and `51 runs planned, 51 pending`
if the grid is right. `sweep.yaml` deliberately does not set `allow_cpu`, so a box whose
driver did not come up aborts in the first second rather than starting a 6-exaFLOP job on a
CPU; if you see that error, go back to step 5.

### Step 10. Check progress

```bash
# where it is in the grid
grep 'exp05.sweep' outputs/05_lm_training/sweep/run.log | tail -5

# the current run's step counter and throughput
tail -3 outputs/05_lm_training/sweep/run.log

# how many runs are finished (target: 51)
find outputs/05_lm_training/sweep -name results.json -path '*seed*' | wc -l

# is the card busy
nvidia-smi
```

The sweep prints `[n/51] track/size/arm/seedK` when it starts each run, and the training
loop prints a `step k/total loss ... bpc {...} (n tok/s)` line at every evaluation, which
is every 25 steps at the 50M size and every 100 at 125M (`sweep.yaml`, `eval_every`).

### Step 11. Finished, or stalled?

**Finished** looks like exactly this, in the log:

```
INFO exp05.sweep | sweep finished: N runs trained this invocation
INFO sanskrit_tok.experiment | wrote .../outputs/05_lm_training/sweep/results.json
INFO exp05.aggregate | wrote .../bpc_curves.pdf, .../bpc_curves.png, .../bpc_final.pdf, .../bpc_final.png
INFO exp05.run | done: .../outputs/05_lm_training/sweep
```

and `find ... -name results.json -path '*seed*' | wc -l` returns **51**.

**Stalled** is: `nvidia-smi` showing 0% utilisation with the Python process still alive,
or no new log line for far longer than one evaluation interval. Judge the interval, do not
guess it: log lines arrive every `eval_every` steps plus one full pass over 3.3 to 3.4 MB of
held-out text, so a gap of several minutes at the 50M size and of ten or twenty minutes at
125M on a slow card is normal. A gap of an hour with the card idle is not. Stalls are
unusual here; the loop makes no network calls and touches no external service once step 6 is
done.

**Slow but healthy** is the common case and looks different: timestamps advancing, a
`step k/total` line arriving every few seconds to few minutes, and utilisation high.
`T7_byt5` is the arm that takes the longest by a wide margin, because at 0.98 bytes per
token the equal-bytes budget hands it four to five times the tokens of a 64k arm
(`README.md`, budget table). Its Track 2 runs are 11,658 steps each. That is not a stall.

If you genuinely need to know how far into a run it is, the step line gives you
`step k/total` directly.

### Step 12. Bring the results home

From your laptop, in the repository root:

```bash
rsync -av <user>@<host>:sanskrit-token/outputs/05_lm_training/sweep/ \
    outputs/05_lm_training/sweep/
```

This is small: each run writes `results.json`, `config.yaml`, `curve.jsonl` and
`sweep_run.json`, which measured 20 KB per run on the smoke sweep, and the sweep level adds
one `results.json` and four figures. `ckpt.pt` is deleted as each run finishes, because
`sweep.yaml` sets `keep_checkpoints: false`; nothing downstream reads a checkpoint, and 51
of them would be roughly 60 GB. Expect tens of MB in total.

Verify locally before you stop paying:

```bash
find outputs/05_lm_training/sweep -name results.json -path '*seed*' | wc -l   # 51
ls outputs/05_lm_training/sweep/results.json \
   outputs/05_lm_training/sweep/bpc_curves.pdf \
   outputs/05_lm_training/sweep/bpc_final.pdf
```

If you want the aggregation redone locally, or you are aggregating a partially finished
sweep:

```bash
uv run python experiments/05_lm_training/aggregate.py \
    --config experiments/05_lm_training/sweep.yaml
```

### Step 13. Destroy the instance

Safe once, and only once, all of these hold:

- the log ended with `done:`, or you have decided to stop early on purpose;
- `find outputs/05_lm_training/sweep -name results.json -path '*seed*' | wc -l` returns 51
  **on your laptop**, not on the box;
- `outputs/05_lm_training/sweep/results.json` and the four figure files are on your laptop;
- you have opened `results.json` and it is not empty.

Then destroy it, and check the dashboard afterwards that it really is gone and that any
attached volume is deleted too. A stopped instance with a persistent volume still bills for
storage; a terminated one does not.

---

## 5. Failure modes, and what each one costs

The single fact behind all of these: **the sweep is resumable at two levels**, and both are
in the code rather than in a promise.

*Between runs.* `is_complete` in `sweep.py` skips a run whose `results.json` exists **and**
whose `sweep_run.json` records a plan hash matching the current config. `pending_runs`
filters the grid through it at startup. So re-running the identical command trains only
what is missing, with no argument changes.

*Within a run.* `train` in `src/sanskrit_tok/lm/train.py` writes `ckpt.pt` at every
evaluation, and `config.resume` defaults to `True`, so a restarted run reloads the model,
the optimiser state, the step counter, the tokens seen, the elapsed seconds, the batch
sampler's RNG state and torch's global generator states, and continues the same stream of
batches. `curve.jsonl` is truncated back to the checkpoint's step so a crash between an
evaluation and a checkpoint cannot leave a duplicate row. `prune_checkpoint` deletes
`ckpt.pt` only *after* `results.json` is written, so an unfinished run always has one.

Checkpoints are written every `eval_every` steps: 25 at the 50M size, 100 at 125M
(`sweep.yaml`). **The most you can lose to an interruption is that many steps of one run**,
which is minutes.

### The connection drops

Nothing is lost, if you started under `tmux` or `nohup` as in step 9: the process is not a
child of your SSH session. Reconnect and `tmux attach -t sweep`, or
`tail -f outputs/05_lm_training/sweep/run.log`.

If you forgot and ran it in a bare shell, the process dies with the session. Reconnect,
re-run the identical command from step 9, and it resumes: finished runs are skipped, the
interrupted one restarts from its last checkpoint. Cost: up to 25 or 100 steps.

### The instance is reclaimed (spot or interruptible)

**This sweep survives interruption of the process. It does not survive loss of the disk.**
That distinction is what decides whether spot pricing is safe for you:

- If the provider preserves the volume across a reclaim, and you start a new instance
  attached to the same volume, re-running the step 9 command resumes exactly as above. You
  lose the reclaimed run's progress since its last checkpoint, plus the minutes to bring a
  new instance up, plus a re-run of `uv sync` if the environment lived on ephemeral disk.
- If the reclaim destroys the disk, you lose **everything not yet copied off**: all
  finished runs' `results.json`, the encoded cache, the uploaded corpora, the environment.
  You start from step 4.

So: if you use spot, put `~/sanskrit-token` on a **persistent volume**, and additionally
`rsync` the results directory back to your laptop every few hours. It is tens of MB and it
converts a total loss into a partial one, because a resumed sweep on a fresh box with those
`results.json` files restored will skip the runs they represent.

The `sweep_run.json` plan hash includes the tokenizer file's sha256, so restored results
are only skipped if the tokenizer they were produced with is the one now present. That is
the intended behaviour and it protects you from resuming onto a different vocabulary.

### The disk fills

Symptom: an `OSError: No space left on device` from a `torch.save` or from the encoder, and
the sweep dies. Nothing already written is corrupted; a truncated `ckpt.pt` will fail to
load on the next attempt, in which case delete that one file and the run restarts from
zero, losing that run only.

Free space and resume:

```bash
df -h .
du -sh outputs/05_lm_training/encoded outputs/05_lm_training/sweep ~/.cache/uv
uv cache prune            # reclaims the download cache; the venv is unaffected
```

You can also delete `outputs/05_lm_training/encoded/*.bin` for arms already finished; they
are regenerated on demand and validated by sha256. Prevention is step 2: **60 GB or more.**

### `uv sync` resolves a torch build without CUDA support

You would see it at step 5 as `torch.cuda.is_available()` returning `False`, and then the
sweep would abort on its first line, because `sweep.yaml` does not set `allow_cpu` and
`resolve_device` raises rather than falling back (`src/sanskrit_tok/lm/config.py`). This
fails loudly and costs you minutes, not a day, which is the point of that design.

In practice `uv sync` on Linux x86_64 resolves the CUDA build: `uv.lock` pins torch 2.14.0
with `nvidia-cudnn-cu13`, `nvidia-nccl-cu13`, `cuda-toolkit` and `triton` all marked
`sys_platform == 'linux'`. The realistic causes of a `False` are, in order of likelihood:

1. **The host driver is too old** for the CUDA 13 wheels. Check `nvidia-smi` for a driver
   of 580 or newer (step 5). Fix: a newer image, or a different provider template.
2. **The container was started without GPU access.** `nvidia-smi` itself fails or reports
   no devices. Fix: this is the provider's launch configuration, not yours; recreate the
   instance from a GPU-enabled template.
3. **An ARM box.** `uv.lock` has aarch64 Linux wheels for torch, but GPU support there is
   image-dependent and this project has not been run on one. Rent x86_64.

Do not "fix" this by `pip install`-ing a different torch. That leaves the environment
disagreeing with `uv.lock`, which is the provenance every `results.json` records.

### The numbers come out looking wrong

Not a failure of the rental, but worth knowing where to look. `results.json` records the
resolved device, the dtype, the hardware, the git commit, the config, the parameter counts,
and per evaluation set the `total_nats`, `n_tokens`, `n_chars_scored` and
`n_chars_denominator` behind every BPC. `sweep_run.json` records the byte budget, the
measured bytes-per-token and the token budget they produced. Almost any surprise can be
diagnosed from those two files without retraining.

---

## 6. Pre-flight checklist

The first five cost nothing. The rest are the gates on the running box: each is minutes of
billing and each of them catches something.

Before you spend anything:

- [ ] `uv run pytest -q` passes on your laptop.
- [ ] `data/processed/lm/` on your laptop contains `track1_raw.txt`, `track1_split.txt`,
      `track2_sample.txt`, 11 `heldout_*.txt` and the manifests.
- [ ] `outputs/tokenizers/` contains the six `_dcs` arms listed in step 6.
- [ ] You have an SSH key, and it is registered with the provider.
- [ ] You know your provider's billing granularity, its minimum credit purchase, and where
      the button that destroys the instance is.

On the box, in this order:

- [ ] It has one GPU, Ampere or newer, 24 GB or more, and `df -h` shows 60 GB or more free.
- [ ] `nvidia-smi` shows a driver of 580 or newer.
- [ ] The torch check from step 5 prints `True ... True`.
- [ ] `rsync` finished, and the box has 863 MB in the right places.
- [ ] The smoke sweep finished and printed `device cuda, dtype bfloat16`.
- [ ] The dry run reported 51 runs and roughly 5,120 PFLOPs of training.
- [ ] You launched under `tmux` or `nohup`, and read the first log line.
- [ ] If on spot: the working directory is on a persistent volume, and you have a plan to
      `rsync` results back periodically.

## 7. You are done when

- [ ] The log ends with `sweep finished`, then the aggregation lines, then `done:`.
- [ ] 51 per-seed `results.json` files exist.
- [ ] `outputs/05_lm_training/sweep/results.json` exists and is non-empty.
- [ ] `bpc_curves.pdf`, `bpc_curves.png`, `bpc_final.pdf`, `bpc_final.png` exist.
- [ ] All of the above are **on your laptop**, copied back with step 12's `rsync`.
- [ ] The instance and any attached volume are destroyed, and the dashboard agrees.
- [ ] The provider's billing page shows what you expected to spend.
