# AI & LLM Infra Blogs

Daily automated aggregator and searchable static dashboard tracking AI systems, LLM inference engines, GPU kernels, cluster orchestration, and frontier lab engineering blogs.

**Live Site:** https://lun0522.github.io/ml-infra-blogs/

## Monitored Sources

33 active AI/LLM infrastructure blogs are configured in [`sources.yaml`](sources.yaml):

| Source | Mode | Endpoint | Focus |
| :--- | :--- | :--- | :--- |
| [Aleksa Gordić](https://www.aleksagordic.com/blog) | `rss` | `https://www.aleksagordic.com/feed.xml` | GPU internals, inference engine breakdowns, large-scale systems analysis |
| [Anthropic Engineering](https://www.anthropic.com/engineering) | `html_links` | `https://www.anthropic.com/engineering` | Claude engineering, harness design, context & agent infrastructure |
| [Anyscale / Ray](https://www.anyscale.com/blog) | `sitemap` | `https://www.anyscale.com/sitemap.xml` | Ray Core/Serve/Train, heterogeneous GPU cluster scheduling |
| [Baseten Blog](https://www.baseten.co/blog/) | `html_links` | `https://www.baseten.co/blog/` | Production inference, TensorRT-LLM, quantization, latency tuning |
| [Cerebras](https://www.cerebras.ai/blog) | `sitemap` | `https://www.cerebras.ai/sitemap.xml` | Wafer-scale inference, disaggregated prefill/decode, Tungsten compiler, MoE tuning |
| [Cloudflare Blog](https://blog.cloudflare.com/) | `rss` | `https://blog.cloudflare.com/rss/` | Edge inference, Workers AI, distributed infrastructure, systems engineering |
| [Crusoe Blog](https://www.crusoe.ai/resources/blog) | `html_links` | `https://www.crusoe.ai/resources/blog` | Bare-metal GPU cloud fabrics, data center power, cluster infra |
| [E2B](https://e2b.dev/blog) | `rss` | `https://e2b.dev/rss.xml` | Agent execution sandboxes, microVM compute scaling, workload isolation |
| [fal](https://blog.fal.ai/) | `rss` | `https://blog.fal.ai/rss/` | Speculative decoding (EAGLE), fast diffusion/video inference and training |
| [Fireworks AI](https://fireworks.ai/blog) | `html_links` | `https://fireworks.ai/blog` | Fast inference engines, speculative decoding, serving architecture |
| [FlashInfer](https://flashinfer.ai/) | `rss` | `https://flashinfer.ai/feed.xml` | Custom GPU kernels for LLM serving, autotuning, MoE and RL kernel optimizations |
| [FriendliAI](https://friendli.ai/blog) | `html_links` | `https://friendli.ai/blog` | Serving throughput, KV/prefix caching, quantized and agentic inference |
| [Gimlet Labs](https://gimletlabs.ai/blog) | `html_links` | `https://gimletlabs.ai/blog` | Agentic kernel verification, multi-silicon inference, disaggregated speculative decoding |
| [humans& Blog](https://humansand.ai/blogs/) | `html_links` | `https://humansand.ai/blogs/` | Engineering and systems notes from humans& |
| [Lambda](https://lambda.com/blog) | `rss` | `https://lambda.com/blog/rss.xml` | GPU clusters, large-scale optical networking (CPO), distributed training kernels |
| [Lilian Weng (Lil'Log)](https://lilianweng.github.io/) | `rss` | `https://lilianweng.github.io/index.xml` | Technical surveys on LLM systems, inference-time compute, agents |
| [LMSYS / SGLang](https://www.lmsys.org/blog/) | `sitemap` | `https://www.lmsys.org/sitemap.xml` | SGLang runtime, RadixAttention, disaggregated serving, systems research |
| [Modal Blog](https://modal.com/blog) | `rss` | `https://modal.com/blog/atom.xml` | Serverless GPU containers, cold-start optimization, inference workloads |
| [Modular Blog](https://www.modular.com/blog) | `rss` | `https://www.modular.com/blog/rss.xml` | MAX engine, Mojo GPU kernels, compiler-driven ML execution |
| [Nebius](https://nebius.com/blog) | `html_links` | `https://nebius.com/blog` | Distributed K8s/NUMA training tuning, RL auto-tuning, MLPerf, world-model serving |
| [Netpreme Blog](https://netpreme.com/blog) | `sitemap` | `https://netpreme.com/sitemap.xml` | Scale-up memory tiers, KV-cache acceleration, sparse attention infrastructure |
| [NVIDIA Data Center & Cloud](https://developer.nvidia.com/blog/category/data-center-cloud/) | `rss` | `https://developer.nvidia.com/blog/category/data-center-cloud/feed/` | CUDA, TensorRT-LLM, NCCL, NVLink/InfiniBand networking, GPU architecture |
| [OpenAI Engineering](https://openai.com/news/engineering/) | `rss` | `https://openai.com/news/engineering/rss.xml` | ChatGPT storage/serving scale, Codex sandboxes, real-time and inference systems |
| [Periodic Labs](https://periodic.com/news) | `html_links` | `https://periodic.com/news` | Scientific AI infrastructure, training, inference, and autonomous labs |
| [Prime Intellect](https://www.primeintellect.ai/blog) | `sitemap` | `https://www.primeintellect.ai/sitemap.xml` | Distributed RL (`prime-rl`), fused MoE kernels, NIXL weight transfer, sandboxes |
| [PyTorch Blog](https://pytorch.org/blog/) | `rss` | `https://pytorch.org/feed/` | `torch.compile`, Triton/Inductor, distributed collectives, PyTorch internals |
| [SemiAnalysis](https://newsletter.semianalysis.com/) | `rss` | `https://newsletter.semianalysis.com/feed` | Chips, accelerator benchmarks, cluster networking, data center infrastructure |
| [SkyPilot Blog](https://skypilot.ai/blog) | `html_links` | `https://skypilot.ai/blog` | Multi-cloud GPU orchestration, spot recovery, distributed job queues |
| [Thinking Machines Blog](https://thinkingmachines.ai/blog/) | `html_links` | `https://thinkingmachines.ai/blog/` | Training systems, numerical determinism, frontier ML infrastructure |
| [Tim Dettmers](https://timdettmers.com/) | `rss` | `https://timdettmers.com/feed/` | GPU hardware analysis, quantization (`bitsandbytes`), memory hierarchies |
| [Together AI](https://www.together.ai/blog) | `rss` | `https://www.together.ai/blog/rss.xml` | Distributed training/inference stacks, FlashAttention/kernels, GPU clusters |
| [Turbopuffer](https://turbopuffer.com/blog) | `rss` | `https://turbopuffer.com/blog/rss.xml` | Vectorized query execution, object-storage search engines, RL retrieval models |
| [vLLM Blog](https://vllm.ai/blog) | `rss` | `https://vllm.ai/blog/rss.xml` | vLLM engine releases, PagedAttention, scheduling, distributed serving |

## How It Works

1. **Declarative Source List ([`sources.yaml`](sources.yaml))**:
   - **`rss` mode**: Standard RSS/Atom/Substack feeds (`aleksagordic.com`, `blog.cloudflare.com`, `blog.fal.ai`, `developer.nvidia.com`, `e2b.dev`, `flashinfer.ai`, `lambda.com`, `lilianweng.github.io`, `modal.com`, `modular.com`, `openai.com/news/engineering`, `pytorch.org`, `semianalysis`, `timdettmers.com`, `together.ai`, `turbopuffer.com`, `vllm.ai`).
   - **`sitemap` mode**: XML sitemaps paired with lazy metadata enrichment only for unseen article URLs (`anyscale.com`, `cerebras.ai`, `lmsys.org`, `netpreme.com`, `primeintellect.ai`).
   - **`html_links` mode**: Plain HTML link extraction + card/article metadata parsing without needing a headless browser (`anthropic.com/engineering`, `baseten.co`, `crusoe.ai`, `fireworks.ai`, `friendli.ai`, `gimletlabs.ai`, `humansand.ai`, `nebius.com`, `periodic.com/news`, `skypilot.ai`, `thinkingmachines.ai/blog/`).
2. **Persistent Archive (`data/posts.json`)**:
   - Preserves `first_seen` timestamps and up to the latest 50 posts per source across runs so posts remain archived even after rolling off a blog's front page, and previously seen non-RSS articles don't need re-fetching.
3. **Static Dashboard (`frontend/`)**:
   - Built with Vite, React, TypeScript, and Tailwind CSS v4 with light/dark mode, paginated timeline view, instant search, and source filtering.

## Running Locally

### Backend (Python 3.11+)

```bash
cd backend
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

# Run unit tests
.venv/bin/pytest tests -v

# Fetch live updates -> data/posts.json + frontend/public/data/posts.json
.venv/bin/python -m app.scripts.update

# Or rebuild frontend/public/data offline from the saved archive
.venv/bin/python -m app.scripts.update --from-archive
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173`. Production builds (`npm run build`) emit static assets into `frontend/dist/`.
