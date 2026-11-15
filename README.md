# AI & LLM Infra Blogs

Daily automated aggregator and searchable static dashboard tracking AI systems, LLM inference engines, GPU kernels, cluster orchestration, and frontier lab engineering blogs.

**Target GitHub Pages URL:** `https://lun0522.github.io/ml-infra-blogs/`

## How It Works

1. **Declarative Source List ([`sources.yaml`](sources.yaml))**:
   - **`rss` mode**: Standard RSS/Atom/Substack feeds (`aleksagordic.com`, `developer.nvidia.com`, `lilianweng.github.io`, `modal.com`, `modular.com`, `pytorch.org`, `semianalysis`, `timdettmers.com`, `together.ai`, `vllm.ai`).
   - **`sitemap` mode**: XML sitemaps paired with lazy metadata enrichment only for unseen article URLs (`anyscale.com`, `lmsys.org`).
   - **`html_links` mode**: Plain HTML link extraction + card/article metadata parsing without needing a headless browser (`anthropic.com/engineering`, `baseten.co`, `crusoe.ai`, `fireworks.ai`, `humansand.ai`, `skypilot.ai`, `thinkingmachines.ai/blog/`).
2. **Persistent Archive (`data/posts.json`)**:
   - Preserves `first_seen` timestamps and up to the latest 50 posts per source across runs so posts remain archived even after rolling off a blog's front page, and previously seen non-RSS articles don't need re-fetching.
3. **Static Dashboard (`frontend/`)**:
   - Built with Vite, React, TypeScript, and Tailwind CSS v4 with light/dark mode, paginated timeline view, instant search, and source filtering.

## Monitored Sources

19 active AI/LLM infrastructure blogs are configured in [`sources.yaml`](sources.yaml):

| Source | Mode | Endpoint | Focus |
| :--- | :--- | :--- | :--- |
| [Aleksa Gordić](https://www.aleksagordic.com/blog) | `rss` | `https://www.aleksagordic.com/feed.xml` | GPU internals, inference engine breakdowns, large-scale systems analysis |
| [Anthropic Engineering](https://www.anthropic.com/engineering) | `html_links` | `https://www.anthropic.com/engineering` | Claude engineering, harness design, context & agent infrastructure |
| [Anyscale / Ray](https://www.anyscale.com/blog) | `sitemap` | `https://www.anyscale.com/sitemap.xml` | Ray Core/Serve/Train, heterogeneous GPU cluster scheduling |
| [Baseten Blog](https://www.baseten.co/blog/) | `html_links` | `https://www.baseten.co/blog/` | Production inference, TensorRT-LLM, quantization, latency tuning |
| [Crusoe Blog](https://www.crusoe.ai/resources/blog) | `html_links` | `https://www.crusoe.ai/resources/blog` | Bare-metal GPU cloud fabrics, data center power, cluster infra |
| [Fireworks AI](https://fireworks.ai/blog) | `html_links` | `https://fireworks.ai/blog` | Fast inference engines, speculative decoding, serving architecture |
| [humans& Blog](https://humansand.ai/blogs/) | `html_links` | `https://humansand.ai/blogs/` | Engineering and systems notes from humans& |
| [Lilian Weng (Lil'Log)](https://lilianweng.github.io/) | `rss` | `https://lilianweng.github.io/index.xml` | Technical surveys on LLM systems, inference-time compute, agents |
| [LMSYS / SGLang](https://www.lmsys.org/blog/) | `sitemap` | `https://www.lmsys.org/sitemap.xml` | SGLang runtime, RadixAttention, disaggregated serving, systems research |
| [Modal Blog](https://modal.com/blog) | `rss` | `https://modal.com/blog/atom.xml` | Serverless GPU containers, cold-start optimization, inference workloads |
| [Modular Blog](https://www.modular.com/blog) | `rss` | `https://www.modular.com/blog/rss.xml` | MAX engine, Mojo GPU kernels, compiler-driven ML execution |
| [NVIDIA Data Center & Cloud](https://developer.nvidia.com/blog/category/data-center-cloud/) | `rss` | `https://developer.nvidia.com/blog/category/data-center-cloud/feed/` | CUDA, TensorRT-LLM, NCCL, NVLink/InfiniBand networking, GPU architecture |
| [PyTorch Blog](https://pytorch.org/blog/) | `rss` | `https://pytorch.org/feed/` | `torch.compile`, Triton/Inductor, distributed collectives, PyTorch internals |
| [SemiAnalysis](https://newsletter.semianalysis.com/) | `rss` | `https://newsletter.semianalysis.com/feed` | Chips, accelerator benchmarks, cluster networking, data center infrastructure |
| [SkyPilot Blog](https://skypilot.ai/blog) | `html_links` | `https://skypilot.ai/blog` | Multi-cloud GPU orchestration, spot recovery, distributed job queues |
| [Thinking Machines Blog](https://thinkingmachines.ai/blog/) | `html_links` | `https://thinkingmachines.ai/blog/` | Training systems, numerical determinism, frontier ML infrastructure |
| [Tim Dettmers](https://timdettmers.com/) | `rss` | `https://timdettmers.com/feed/` | GPU hardware analysis, quantization (`bitsandbytes`), memory hierarchies |
| [Together AI](https://www.together.ai/blog) | `rss` | `https://www.together.ai/blog/rss.xml` | Distributed training/inference stacks, FlashAttention/kernels, GPU clusters |
| [vLLM Blog](https://vllm.ai/blog) | `rss` | `https://vllm.ai/blog/rss.xml` | vLLM engine releases, PagedAttention, scheduling, distributed serving |

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

## Deploying to GitHub Pages

1. Push to `main`.
2. In your GitHub repository: **Settings → Pages → Build and deployment → Source**, choose **GitHub Actions**.
3. Also under **Settings → Actions → General → Workflow permissions**, ensure **Read and write permissions** is enabled (needed for the daily job to commit `data/posts.json` updates).
4. Trigger **Daily Blog Monitor & GitHub Pages Deploy** (`.github/workflows/deploy.yml`) manually from the **Actions** tab, or let it run daily at `14:00 UTC`.
