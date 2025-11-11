# CacheProbe: Auditing Prompt Cache Isolation in Inference & Gateway APIs

The goal of this research project is to investigate whether OpenRouter's API gateway architecture introduces prompt caching vulnerabilities that bypass provider-level prompt cache isolation guarantees. LLM providers (should) implement per-account or per-organization prompt caching to prevent timing attacks, but does routing through OpenRouter with shared organizational credentials inadvertently creates global cache sharing across all OpenRouter users?

## Usage

TODO

## References

This project is heavily inspired by the original research paper [Auditing Prompt Caching in Language Model APIs](https://arxiv.org/pdf/2502.07776) by Chenchen Gu, Xiang Lisa Li, Rohith Kuditipudi, Percy Liang, and Tatsunori Hashimoto.
