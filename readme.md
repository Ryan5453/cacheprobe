# CacheProbe: Auditing Prompt Cache Isolation in Gateway APIs

The goal of this research project is to investigate whether OpenRouter's API gateway architecture introduces prompt caching vulnerabilities that bypass provider-level prompt cache isolation guarantees. LLM providers (should) implement per-account or per-organization prompt caching to prevent timing attacks, but does routing through OpenRouter with shared organizational credentials inadvertently creates global cache sharing across all OpenRouter users?

## Setup

Create a `config.yaml` file with your API keys and provider configurations:

```yaml
openrouter:
  keys:
    - "your-openrouter-api-key-1"
    - "your-openrouter-api-key-2"  # For cross-account tests

vercel:
  keys:
    - "your-ai-gateway-api-key-1"
    - "your-ai-gateway-api-key-2"  # For cross-account tests

groq:
  keys:
    - "your-groq-api-key-1"
    - "your-groq-api-key-2"  # For cross-account tests
  direct:
    model: "llama-3.3-70b-versatile"
    base_url: "https://api.groq.com/openai/v1"
  openrouter:
    model: "groq/llama-3.3-70b-versatile"
  vercel:
    model: "groq/llama-3.3-70b-versatile"
  num_samples: 250
  num_victim_requests: 1
  prompt_length: 4096
  prefix_fraction: 0.95
  delay_between_requests: 0.5
  max_retries: 5
  use_cache_keys: false
```


## Usage

Run audits using the command line:

```bash
# Run all tests (direct, OpenRouter, and Vercel for providers with vercel config)
uv run main.py --mode all

# Run a single test for a specific provider and scenario
uv run main.py --mode single --provider groq --scenario vercel_same_account
uv run main.py --mode single --provider groq --scenario direct_same_account
uv run main.py --mode single --provider groq --scenario openrouter_default_cross_account

# Run BYOK tests only (view note below)
uv run main.py --mode byok
```

Results are saved to the `results/` directory by default.

> [!NOTE]
> OpenRouter BYOK tests are separated into their own method due to the need to need to manually disable/enable BYOK on the OpenRouter dashboard. It is highly recommended to enable "Always use for this provider" to avoid accidental fallback to OpenRouter's keys.


## References

This project is heavily inspired by the original research paper [Auditing Prompt Caching in Language Model APIs](https://arxiv.org/pdf/2502.07776) by Chenchen Gu, Xiang Lisa Li, Rohith Kuditipudi, Percy Liang, and Tatsunori Hashimoto.

