# CacheProbe: Auditing Prompt Cache Isolation in Inference & Gateway APIs

The goal of this research project is to investigate whether OpenRouter's API gateway architecture introduces prompt caching vulnerabilities that bypass provider-level prompt cache isolation guarantees. LLM providers (should) implement per-account or per-organization prompt caching to prevent timing attacks, but does routing through OpenRouter with shared organizational credentials inadvertently creates global cache sharing across all OpenRouter users?

## Setup

Create a `config.yaml` file with your API keys and provider configurations:

```yaml
groq:
  keys:
    victim: "your-victim-api-key"
    attacker: "your-attacker-api-key"
  direct:
    model: "openai/gpt-oss-20b"
    base_url: "https://api.groq.com/openai/v1"
  openrouter:
    model: "openai/gpt-oss-20b"
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
# This will test prompt caching for the same account for the Groq API
uv run main.py --mode single --provider groq --scenario direct_same_account

# This will test prompt caching isolation cross-account for the OpenRouter API
uv run main.py --mode single --provider openrouter --scenario openrouter_default_cross_account

# Run all non-BYOK tests
uv run main.py --mode all

# Run BYOK tests only (view note below)
uv run main.py --mode byok
```

Results are saved to the `results/` directory by default.

> [!NOTE]
> OpenRouter BYOK tests are separated into their own method due to the need to need to manually disable/enable BYOK on the OpenRouter dashboard. It is highly recommended to enable "Always use for this provider" to avoid accidental fallback to OpenRouter's keys.


## References

This project is heavily inspired by the original research paper [Auditing Prompt Caching in Language Model APIs](https://arxiv.org/pdf/2502.07776) by Chenchen Gu, Xiang Lisa Li, Rohith Kuditipudi, Percy Liang, and Tatsunori Hashimoto.
