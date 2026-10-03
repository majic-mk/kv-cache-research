# Approval update — 2026-10-03 08:38 UTC

The user approved a **cumulative CNY100 ceiling** for AutoDL server rental for this research project; spending beyond that requires a new approval. Cloud-browser operation is authorized. The first pilot remains internally capped at four GPU hours / CNY20 before evidence review. No new top-up, automatic recharge, paid disk expansion, or new legal agreement is authorized by this record. Account verification, actual instance quote, code-transfer readiness, and functioning stop controls remain prerequisites. No rental or GPU execution has occurred.

The dated estimates below are the planning basis. The numerical ceiling is now approved; execution readiness and evidence gates remain mandatory.

# GPU provider research and proposed exploratory budget

Checked: 2026-10-03 UTC; AutoDL pricing corrected after direct official frontend-asset verification at approximately 07:33 UTC. Status: **cumulative CNY100 ceiling approved at 08:38 UTC; execution readiness pending**. No rental, login, account creation, payment, model download to a paid GPU, or GPU experiment has been performed for this research.

## Current sequence and recommendation

Finish cloud-based topic selection, code preparation, and CPU tests first. GPU execution is deferred until cloud-browser AutoDL account and control readiness are verified. An earlier computer-access attempt did not establish a usable remote executor; do not assume that any assistant can currently execute commands or shut down an AutoDL instance.

The leading candidate is **one AutoDL ordinary pay-as-you-go container with an RTX 5090 32GB**, conditional on the account's actual listing and compatible software. Proposed gate A: **at most 4 billable GPU hours and CNY 20 total**, whichever limit is reached first. This includes provisioning, environment setup, downloads while the GPU instance is running, inference, and any ancillary fees. No disk expansion or automatic renewal/recharge is proposed. Proposed total for early exploration: **CNY 100 and 28 GPU hours** across separately gated stages. This is an exploratory budget, **not a paper-completion budget** or prediction that full reproduction fits in 28 hours.

## Evidence and provider comparison

All prices below are public list/starting prices, not an account-specific checkout quote or a reservation. Availability, regional eligibility, account balance, tax, currency/payment fees, and final order terms must be checked at execution time. An earlier indexed AutoDL homepage result exposed the membership price and was incorrectly labeled as an ordinary-user price; the directly retrieved official frontend asset below supersedes that interpretation. No account inventory was inspected.

### AutoDL: recommended candidate, subject to quote verification

- **Corrected public price: RTX 5090 / 32GB original/ordinary-user price is CNY 2.93 per hour.** The official frontend data specifies originalPrice 2.93, a normal_user discount factor of 1, and a member_level_3 discount factor of 0.95 for GPU name 5090. The member calculation is 2.93 × 0.95 = CNY 2.7835/hour, approximately **CNY 2.78/hour** as shown in the indexed page. Membership must not be assumed. [Official frontend pricing data](https://www.autodl.com/assets/index.ad3fe4aa.js), [homepage](https://www.autodl.com/home)
- Evidence retrieved independently on 2026-10-03 at approximately 07:33 UTC: public asset index.ad3fe4aa.js, 19,272 bytes, SHA-256 `6fec1d1fa2951f016cc3499a49b64d56162ca0f55a3dc01a02e4c57e44b2bc63`. Data fields verified: RTX 5090, 32GB, originalPrice 2.93; member_level_3 / 5090 / discount 0.95; normal_user / 5090 / discount 1. This establishes public frontend prices only; actual selected-machine quote remains unknown. The asset retrieval date is not a claimed price-publication date.
- Ordinary pay-as-you-go compute runs from power-on to power-off, calculated to the second with **CNY 0.01 minimum charge**. Idle-but-running instances are charged. GPU capacity is not reserved after shutdown. [Billing](https://api.autodl.com/docs/price/)
- Ordinary-container documentation describes a **30GB system disk** and **50GB free data disk**. Paid data-disk expansion is separate; its reference member rate is about **CNY 0.0066 per GB per day**, based on daily peak allocated capacity, including stopped days until released/shrunk. These are not promises for every product variant. [Filesystem](https://www.autodl.com/docs/env/), [data disk](https://api.autodl.com/docs/local_disk/)
- **Container Instance Pro differs:** default 30GB system disk has a CNY 0.10/day minimum while the instance exists and has no separate data disk. Do not apply ordinary-container free-disk assumptions to Pro. [Pro data rules](https://www.autodl.com/docs/instance_pro_data/)
- CPU and host RAM are listing-dependent and were **not verified for a specific CNY 2.93 ordinary-user listing**. Target at least 8 vCPU and preferably 64GB RAM, 1 GPU with actual 32GB VRAM; choose one model at a time so the free 50GB data disk can be tested first. A lower-RAM listing needs an explicit memory check, not silent substitution.
- No-card mode is publicly documented as 0.5 CPU core, 2GB RAM, CNY 0.10/hour. It can reduce costs for downloads and light preparation, but is not suitable for memory-heavy preprocessing; it remains paid and requires authorization. [Save-money guide](https://www.autodl.com/docs/save_money/)
- Provider documentation explicitly mentions **scheduled shutdown in the console** and an in-container **/usr/bin/shutdown** command. This verifies product support, not a configured or tested stop mechanism for this project. [Scheduled-shutdown mention](https://www.autodl.com/docs/migrate_instance/), [shutdown command](https://www.autodl.com/docs/save_money/)

### Runpod: documented fallback

- Main product table lists **RTX 5090, 32GB VRAM, 35GB host RAM, 9 vCPU, USD 0.99/hour**. Storage is additional/configurable. [Product table](https://www.runpod.io/product/cloud-gpus)
- Another official GPU page shows Community USD 0.69/hour and Secure USD 0.89/hour starting prices. Public pages disagree; use the selected account offer rather than the lowest banner. [5090 page](https://www.runpod.io/gpu/5090)
- Compute is billed per second. Deployment needs at least one hour's configuration cost in credits. The billing page says a new user can start with USD 10; purchased credits are nonrefundable. This is not proof that the user's payment method will accept that amount.
- Container/volume storage while running: USD 0.10/GB/month; stopped volume disk: USD 0.20/GB/month; network volumes below 1TB: USD 0.07/GB/month. Network-volume billing is listed hourly. A stopped 100GB volume disk therefore costs about USD 20/month. Low balance is not a safe budget controller because data may be terminated. [Billing and refunds](https://docs.runpod.io/accounts-billing/billing)
- The listed 35GB RAM is tighter than the preferred 64GB. Actual stock, selected location, disk configuration, sales tax and restart availability remain unverified.

### Beam: attractive published complete machine shape, unresolved checkout

- Public on-demand price: **RTX 5090 32GB, 12 vCPU, 64GB RAM, 500GB NVMe, from USD 0.72/hour**, with CPU/RAM/NVMe included in the machine price. [Official pricing](https://www.beam.cloud/pricing)
- Do not confuse this with the serverless 5090 price: USD 0.000303/second for GPU, additional CPU/RAM, with committed-spend conditions displayed.
- Specific on-demand billing minimum, initial funding minimum, stopped-machine/storage policy, tax and available inventory were not established. Do not rent based on the public headline alone.

### Vast.ai: quote-only fallback

The official dynamic page did not expose a verifiable concrete 5090 offer price during this pass. It is a host-priced marketplace: compute is billed per second and storage continuously, including stopped instances. CPU/RAM/disk/network charges and reliability are offer-specific. Do not turn third-party lowest-price trackers into an actual quote. [Pricing](https://vast.ai/pricing), [official billing FAQ](https://github.com/vast-ai/docs/blob/main/guides/reference/faq/billing.mdx)

## Internal staged budget within the approved CNY100 ceiling

| Gate | Purpose | Maximum additional GPU hours | Maximum additional all-in cost | Release condition |
|---|---|---:|---:|---|
| A | Environment/SM120 compatibility, one model, 32–64-sample smoke test | 4 | CNY 20 | Actual quote and approved cloud execution/stop route verified; cumulative budget respected |
| B | Full-KV plus the selected strong baseline on the agreed evaluation slice | 12 | CNY 40 | A passed; trustworthy quality/memory/time measurements; internal evidence review supports continuation within the approved cumulative ceiling |
| C | Two tiny research variants and a repeat | 12 | CNY 40 | B reproduced sufficiently; candidate merits testing; internal evidence review supports continuation within the approved cumulative ceiling |
| Total | Early feasibility exploration, not a paper budget | 28 | CNY 100 | All gates are proposals; no automatic transition |

At the corrected public ordinary-user rate of CNY 2.93/hour, compute-only estimates are A = CNY 11.72, B = CNY 35.16, C = CNY 35.16, total = CNY 82.04. These calculations do not assume membership; the approximate CNY 2.78/hour membership price is not used in the budget. The remaining allowance is headroom, not permission for add-ons. An optional proposed compute-rate ceiling is CNY 3.00/hour, retained as the operational quote ceiling. Taxes, service/payment/currency fees and any unavoidable attributable storage must fit inside each all-in cap. If the minimum required account top-up exceeds the approved cash-outlay limit, pause rather than depositing more. Existing balance does not eliminate the need for spending authorization.

Gate A permits **no paid disk expansion**. If ordinary free storage is insufficient, stop and resize the task or request a revised quote and authorization. No new recurring storage, annual/monthly reservation, auto-recharge, API bill, additional GPU, or provider substitution is covered by this proposal.

## Operational stop design and honest feasibility status

Nothing below is installed or tested. A local Python timeout or a killed inference process does **not** stop a paid GPU instance. A plan that cannot verify a provider-level stop path must not begin an unattended paid run.

1. Before the GPU phase, verify an authorized control path to the actual instance and console. Record the instance ID, selected GPU/VRAM, host CPU/RAM, hourly quote, billing start time, storage fees and provider time zone.
2. Immediately after provisioning, set the provider's scheduled shutdown and verify it in the console **before workload execution**. Proposed initial deadline: no later than 3h55m from billable start, leaving a five-minute operational margin within the 4h maximum. If configuration cannot be verified promptly, shut down manually; do not continue with a merely promised timer.
3. Compute the earlier cost stop using the actual rate and conservative ancillary-fee reserve. Stop before either time or cost cap; proposed alert/stop trigger is 95% of the all-in cap, rather than waiting until the last cent. Account billing can lag, so elapsed-time accounting is also required.
4. Add a job-level cleanup path that flushes results/logs and calls the provider-supported shutdown on both success and failure. Test the path on the authorized instance. AutoDL's documented command is /usr/bin/shutdown; a success-only shell `&& shutdown` is insufficient.
5. Proposed watchdog: stop after ten minutes with no useful job and no observable download/compile/preprocessing progress. GPU utilization alone is not a reliable idle signal. A watchdog is secondary to the platform timer and is not presently implemented.
6. If model loading plus one inference is still not successful after 45 minutes of paid compatibility work, save diagnostics and stop the GPU. Permit at most one targeted retry of the same OOM/environment error, not an unbounded rebuild loop.
7. After a stage exits, verify provider state shows stopped and review the bill. Export code, configuration, logs and results through the authorized route. Do not assume stopped disks are free or indefinitely retained; releasing/deleting persistent data requires the appropriate user authorization and verified backups.
8. Do not enable auto-recharge or rely on a zero balance to stop compute. Do not leave a GPU on while discussing next steps, reading papers, or waiting for approval. Move those activities back to CPU/cloud preparation.

## Software compatibility and model fit

RTX 5090 has 32GB VRAM and CUDA compute capability **12.0 (SM120)**. [NVIDIA specifications](https://www.nvidia.com/en-gb/geforce/graphics-cards/50-series/rtx-5090/), [compute capability](https://developer.nvidia.com/cuda/gpus)

PyTorch 2.7 introduced Blackwell support with CUDA 12.8 wheels. Treat this as a documented support baseline, not a mandate to use that old version today. Current vLLM stable installation documentation defaults to CUDA 12.9 and also describes other CUDA wheel variants. Fix a compatible model/Transformers/PyTorch/vLLM/attention-backend combination and container digest for the reproduction; do not blindly copy CUDA 11/12.1-era paper environments. [PyTorch release](https://pytorch.org/blog/pytorch-2-7/), [vLLM installation](https://docs.vllm.ai/en/stable/getting_started/installation/gpu/)

SM120 consumer Blackwell is not interchangeable with B200/SM100. vLLM's current implementation distinguishes attention paths, including SM12x decode versus prefill support. Backend-specific FlashAttention, FP8 and FP4 support must pass the actual smoke test. [vLLM implementation documentation](https://docs.vllm.ai/en/stable/api/vllm/utils/flashinfer/)

A 7–8B BF16 model has roughly 14–16GB of weight data before KV, activations, allocator overhead and temporary tensors. Single-GPU inference at batch 1, first 4K/8K then 16K/32K, is a reasonable pilot hypothesis; 128K full-KV, high concurrency, and full-parameter training are not promised. Keep model weights BF16 initially so weight-quantization effects do not confound KV experiments.

Illustrative derived calculation, not a GPU measurement: Qwen2.5-7B-Instruct has 28 layers, 4 KV heads, head dimension 128. At BF16, 2 (K/V) × 28 × 4 × 128 × 2 bytes = **56KiB per cached token**. Thus 8192 tokens use about **448MiB of raw KV**, and 32768 about **1.75GiB**, at batch 1. Real peak memory is larger. Algorithms materializing attention scores can add substantial temporary memory. [Official model configuration](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct/blob/main/config.json)

## Account, activation, payment, and mandatory agreements

- Service delivery is a remotely accessed Linux GPU container in the user's selected AutoDL account; the approved cloud browser is the planned control interface, not the GPU itself. Exact account, region, instance, access method and authority to run/stop it are unresolved.
- AutoDL's public registration form requests a mobile number, verification code and password, and shows acceptance of the Service Agreement, Privacy Policy and Anti-Mining Agreement. No account registration is authorized by this research. Password creation/entry and sensitive identity/payment steps need the supported user handoff. [Registration page](https://autodl.com/register)
- Verified official terms: [AutoDL Service Agreement](https://www.autodl.com/docs/agreements/), [Privacy Policy](https://www.autodl.com/docs/privacy_policy/), [Anti-Mining Agreement](https://www.autodl.com/docs/anti_mining/). Actual checkout may present additional terms; inspect and disclose them before approval. Do not claim these agreements have been accepted.
- Official billing documentation currently lists WeChat/Alipay top-up and corporate remittance, with PayPal described as not yet available there. The minimum top-up, this user's available payment method, usable balance, refunds/withdrawals for ordinary prepaid balance and actual taxes/fees were **not verified**. Reserved-plan conversion refunds do not establish a general cash refund right. [Billing/top-up](https://www.autodl.com/docs/price/)
- Real-name requirements depend on the selected product/feature. The **custom public-service/port exposure** feature explicitly requires real-name verification and a separate supplement; it is unnecessary for a private offline pilot and is excluded. Do not assume all ordinary containers require that feature. [Custom-service supplement](https://www.autodl.com/docs/service_agreement/), [identity-verification notice](https://api.autodl.com/docs/real_name_cert/)
- No new API key or ongoing credential access is part of the budget proposal. Do not save credentials in this project.

## Original gate-A approval wording (historical, superseded by the CNY100 approval)

This historical draft is superseded; any new order-level clarification should occur only after cloud topic/code work is ready, the approved control route is verified, and bracketed account/order fields are actually known. If a material field remains unknown, gather it before asking; this draft does not itself authorize any action.

> 下一步拟在你的 AutoDL 账号［已核验账号］租一台普通按量容器：1 张 RTX 5090 32GB，［CPU/RAM］，使用自带免费磁盘。当前账号报价为 ¥［实际费率］/小时，按秒计费；本轮包括开机准备、计算及已列明杂费在内，最多 ¥20、最多 4 个计费 GPU 小时，先达到任一限制就结束，不扩容磁盘。费用从［已核验的现有余额／待你支付的具体充值金额及方式］扣除，不开自动充值或包月。服务器远程提供在［地区］，后续用你的 Windows 连接；已核验的停机安排是［具体控制台定时关机时间与备用关机方式］。使用需遵守［实际需新接受的协议链接］；［页面显示的退款或取消限制］。你同意这次最多 ¥20 的试跑，以及按上述限制结束并关机吗？

Do not state that a shutdown is configured until it is. If account creation, fresh terms, a CAPTCHA, additional data transfer, larger minimum deposit, or credential configuration is required, include that specific action and consequence in the future request or handoff; do not infer permission from this draft. The full CNY 100 exploration allowance remains a separate proposal.

## Codex/ChatGPT quota boundary

ChatGPT/Codex product allowance is not AutoDL/Runpod GPU credit and is not freely spendable API credit. API-key use follows separate API pricing and available models depend on that key's access. No user subscription balance or API entitlement was inspected, and no paid external model API is planned for this initial pilot. [Official pricing and usage documentation](https://learn.chatgpt.com/docs/pricing)
