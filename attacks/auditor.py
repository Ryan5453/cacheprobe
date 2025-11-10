import random
import string
import time
from enum import Enum

import numpy as np
from openai import OpenAI, RateLimitError
from scipy import stats
from sklearn.metrics import auc, precision_recall_curve


class ScenarioType(str, Enum):
    DIRECT_SAME = "direct_same_account"
    DIRECT_CROSS = "direct_cross_account"
    OR_DEFAULT_SAME = "openrouter_default_same_account"
    OR_DEFAULT_CROSS = "openrouter_default_cross_account"
    OR_BYOK_SAME = "openrouter_byok_same_account"
    OR_BYOK_CROSS = "openrouter_byok_cross_account"


class CachingAuditor:
    def __init__(
        self,
        provider_name: str,
        config: dict,
        api_key: str,
        scenario: ScenarioType,
    ):
        """
        Initialize the CachingAuditor with provider configuration and scenario settings.

        :param provider_name: The name of the LLM provider to audit
        :param config: Configuration settings for the provider
        :param api_key: API key for authentication
        :param scenario: The testing scenario to execute
        """
        self.provider_name = provider_name
        self.config = config
        self.scenario = scenario
        self.prompt_length = config["prompt_length"]
        self.prefix_fraction = config["prefix_fraction"]
        self.delay_between_requests = config.get("delay_between_requests", 0.5)
        self.max_retries = config.get("max_retries", 5)

        if scenario.value.startswith("openrouter"):
            self.client = OpenAI(
                base_url="https://openrouter.ai/api/v1", api_key=api_key
            )
            self.model = config["openrouter"]["model"]
            self.use_provider_control = True
        else:
            self.client = OpenAI(base_url=config["direct"]["base_url"], api_key=api_key)
            self.model = config["direct"]["model"]
            self.use_provider_control = False

    def generate_random_prompt(self, length: int) -> str:
        """
        Generate a prompt containing {length} tokens

        Functionally, this works as generating length amount of tokens
        most tokenizers have every letter prefixed with a space as
        a valid token.

        :param length: Number of tokens to generate
        :return: String of random tokens
        """
        tokens = [random.choice(string.ascii_letters) for _ in range(length)]
        return " ".join(tokens)

    def generate_prefix_prompt(self, base_prompt: str, prefix_fraction: float) -> str:
        """
        Generate prompt with same prefix but different suffix.

        :param base_prompt: Original prompt to use as base
        :param prefix_fraction: Fraction of prompt to keep as prefix (0.0-1.0)
        :return: New prompt with matching prefix and random suffix
        """
        tokens = base_prompt.split()
        prefix_length = int(len(tokens) * prefix_fraction)
        prefix = tokens[:prefix_length]

        suffix_length = len(tokens) - prefix_length
        new_suffix = [random.choice(string.ascii_letters) for _ in range(suffix_length)]

        return " ".join(prefix + new_suffix)

    def measure_ttft(self, prompt: str) -> float:
        """
        Send prompt and measure Time To First Token with retry logic.

        :param prompt: The prompt to send to the Inference API
        :return: Time to first token in seconds
        """
        kwargs = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 1,
        }

        # By default, OpenRouter load balances across all providers.
        # This forces it to use the specified provider so we can test prompt
        # caching isolation for a specific provider.
        if self.use_provider_control:
            kwargs["extra_body"] = {
                "provider": {
                    "order": [self.provider_name],
                    "allow_fallbacks": False,
                }
            }

        for attempt in range(self.max_retries):
            try:
                start_time = time.time()
                self.client.chat.completions.with_raw_response.create(**kwargs)
                end_time = time.time()
                return end_time - start_time
            except RateLimitError as e:
                if attempt == self.max_retries - 1:
                    raise

                retry_after = e.response.headers.get("retry-after")
                wait_time = float(retry_after) + 1 if retry_after else 2**attempt
                time.sleep(wait_time)

        return 0.0

    def cache_miss_procedure(self) -> list[float]:
        """
        Generate cache miss times by sending random prompts.

        :return: List of response times for cache miss scenarios
        """
        miss_times = []

        num_samples = self.config["num_samples"]
        for i in range(num_samples):
            prompt = self.generate_random_prompt(self.prompt_length)
            timing = self.measure_ttft(prompt)
            miss_times.append(timing)

            if i < num_samples - 1:
                time.sleep(self.delay_between_requests)

        return miss_times

    def cache_hit_procedure(self) -> list[float]:
        """
        Attempt to generate cache hit times by reusing prompts.

        :return: List of response times for cache hit scenarios
        """
        hit_times = []

        num_samples = self.config["num_samples"]
        num_victim_requests = self.config["num_victim_requests"]
        for i in range(num_samples):
            base_prompt = self.generate_random_prompt(self.prompt_length)

            for j in range(num_victim_requests):
                self.measure_ttft(base_prompt)
                if j < num_victim_requests - 1:
                    time.sleep(self.delay_between_requests)

            if self.prefix_fraction == 1.0:
                test_prompt = base_prompt
            else:
                test_prompt = self.generate_prefix_prompt(
                    base_prompt, self.prefix_fraction
                )

            timing = self.measure_ttft(test_prompt)
            hit_times.append(timing)

            if i < num_samples - 1:
                time.sleep(self.delay_between_requests)

        return hit_times

    @staticmethod
    def compute_ks_test(
        hit_times: list[float], miss_times: list[float]
    ) -> tuple[float, float]:
        """
        Run Kolmogorov-Smirnov test to compare timing distributions.

        :param hit_times: Response times for cache hit scenarios
        :param miss_times: Response times for cache miss scenarios
        :return: Tuple of (KS statistic, p-value)
        """
        statistic, pvalue = stats.ks_2samp(hit_times, miss_times)
        return statistic, pvalue

    @staticmethod
    def compute_metrics(hit_times: list[float], miss_times: list[float]) -> dict:
        """
        Compute evaluation metrics for cache detection performance.

        :param hit_times: Response times for cache hit scenarios
        :param miss_times: Response times for cache miss scenarios
        :return: Dict with statistical measures
        """
        all_times = hit_times + miss_times
        labels = [1] * len(hit_times) + [0] * len(miss_times)

        scores = [-t for t in all_times]

        precision, recall, _ = precision_recall_curve(labels, scores)
        avg_precision = auc(recall, precision)

        return {
            "avg_precision": float(avg_precision),
            "mean_hit_time": float(np.mean(hit_times)),
            "mean_miss_time": float(np.mean(miss_times)),
            "std_hit_time": float(np.std(hit_times)),
            "std_miss_time": float(np.std(miss_times)),
            "median_hit_time": float(np.median(hit_times)),
            "median_miss_time": float(np.median(miss_times)),
        }

    def run_audit(self) -> dict:
        """
        Run complete audit and return results.

        :return: Dict containing all test data and metrics
        """
        miss_times = self.cache_miss_procedure()
        hit_times = self.cache_hit_procedure()

        ks_statistic, p_value = self.compute_ks_test(hit_times, miss_times)

        metrics = self.compute_metrics(hit_times, miss_times)

        result = {
            "miss_times": miss_times,
            "hit_times": hit_times,
            "ks_statistic": float(ks_statistic),
            "p_value": float(p_value),
            "detected_caching": bool(p_value < 1e-8),
            "metrics": metrics,
        }

        return result
