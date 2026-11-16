export interface PinnedReference {
  id: string;
  title: string;
  author: string;
  url: string;
  summary: string;
}

export const PINNED_REFERENCES: PinnedReference[] = [
  {
    id: "jax-scaling-book",
    title: "How To Scale Your Model",
    author: "Google DeepMind",
    url: "https://jax-ml.github.io/scaling-book/",
    summary:
      "Systems guide to scaling LLMs on TPUs and GPUs covering roofline modeling, transformer FLOPs and memory math, collectives, parallelism strategies, and JAX profiling.",
  },
  {
    id: "designing-ai-chip",
    title: "Designing AI Chip Hardware and Software",
    author: "Bjarke Hammersholt Roune",
    url: "https://docs.google.com/document/d/1dZ3vF8GE8_gx6tl52sOaUVEPq0ybmai1xvu3uk89_is/edit?tab=t.0#heading=h.rduzhxi11vcn",
    summary:
      "Former Google TPUv3 software lead and Nvidia engineer shares AI chip hardware/software trade-offs, advocating cache-and-systolic-array AI CPUs for peak tokens per dollar.",
  },
];
