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

    def measure_ttft(self, prompt: str) -> tuple[float, dict | None]:
        """
        Send prompt and measure Time To First Token with retry logic.

        :param prompt: The prompt to send to the Inference API
        :return: Tuple of (time to first token in seconds, usage dict or None)
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
                raw_response = self.client.chat.completions.with_raw_response.create(
                    **kwargs
                )
                end_time = time.time()

                completion = raw_response.parse()
                usage = completion.usage.model_dump() if completion.usage else None

                return end_time - start_time, usage
            except RateLimitError as e:
                if attempt == self.max_retries - 1:
                    raise

                retry_after = e.response.headers.get("retry-after")
                wait_time = float(retry_after) + 1 if retry_after else 2**attempt
                time.sleep(wait_time)

        return 0.0, None

    def interleaved_procedure(
        self,
    ) -> tuple[list[float], list[float], list[dict | None], list[dict | None]]:
        """
        Run interleaved miss/hit tests to reduce temporal variance.

        This method alternates between cache miss and cache hit tests,
        ensuring each pair is tested under similar conditions (API load,
        network conditions, etc.). This reduces systematic differences
        that could be mistaken for caching effects.

        :return: Tuple of (miss_times, hit_times, miss_usage, hit_usage)
        """
        miss_times = []
        hit_times = []
        miss_usage = []
        hit_usage = []

        num_samples = self.config["num_samples"]
        num_victim_requests = self.config["num_victim_requests"]

        for i in range(num_samples):
            if (i + 1) % 50 == 0:
                print(f"  Progress: {i + 1}/{num_samples} pairs completed")

            miss_prompt = self.generate_random_prompt(self.prompt_length)
            miss_timing, miss_usage_data = self.measure_ttft(miss_prompt)
            miss_times.append(miss_timing)
            miss_usage.append(miss_usage_data)

            time.sleep(self.delay_between_requests)

            hit_base_prompt = self.generate_random_prompt(self.prompt_length)

            for j in range(num_victim_requests):
                self.measure_ttft(hit_base_prompt)
                if j < num_victim_requests - 1:
                    time.sleep(self.delay_between_requests)

            if self.prefix_fraction == 1.0:
                hit_test_prompt = hit_base_prompt
            else:
                hit_test_prompt = self.generate_prefix_prompt(
                    hit_base_prompt, self.prefix_fraction
                )

            hit_timing, hit_usage_data = self.measure_ttft(hit_test_prompt)
            hit_times.append(hit_timing)
            hit_usage.append(hit_usage_data)

            if i < num_samples - 1:
                time.sleep(self.delay_between_requests)

        return miss_times, hit_times, miss_usage, hit_usage

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

    def analyze_cached_tokens(
        self, hit_usage: list[dict | None], miss_usage: list[dict | None]
    ) -> dict:
        """
        Analyze cached tokens in API responses to detect cache usage.
        Only counts as a cache hit if the cached tokens match the expected
        prefix fraction (e.g., 95% or 100% of the prefix).

        :param hit_usage: Usage data for cache hit scenarios
        :param miss_usage: Usage data for cache miss scenarios
        :return: Dict with cache token statistics
        """

        def extract_cached_tokens(usage_list: list[dict | None]) -> list[int]:
            cached = []
            for usage in usage_list:
                if usage and "prompt_tokens_details" in usage:
                    details = usage["prompt_tokens_details"]
                    if isinstance(details, dict) and "cached_tokens" in details:
                        cached.append(details["cached_tokens"])
                    elif hasattr(details, "cached_tokens"):
                        cached.append(details.cached_tokens)
            return cached

        hit_cached = extract_cached_tokens(hit_usage)
        miss_cached = extract_cached_tokens(miss_usage)

        has_cache_data = len(hit_cached) > 0 or len(miss_cached) > 0

        result = {
            "has_cache_data": has_cache_data,
            "hit_samples_with_cache_data": len(hit_cached),
            "miss_samples_with_cache_data": len(miss_cached),
        }

        if has_cache_data:
            threshold = int(self.prompt_length * self.prefix_fraction * 0.9)

            hit_with_cache = sum(1 for c in hit_cached if c >= threshold)
            miss_with_cache = sum(1 for c in miss_cached if c >= threshold)

            result.update(
                {
                    "hit_cache_percentage": (hit_with_cache / len(hit_cached) * 100)
                    if hit_cached
                    else 0,
                    "miss_cache_percentage": (miss_with_cache / len(miss_cached) * 100)
                    if miss_cached
                    else 0,
                }
            )

        return result

    def filter_outliers_automatic(
        self, times: list[float], method: str = "iqr", iqr_multiplier: float = 1.5
    ) -> tuple[list[float], int, float]:
        """
        Automatically filter outliers using statistical methods.
        This uses IQR with a multiplier of 1.5 for standard outliers.

        :param times: List of timing measurements
        :return: Tuple of (filtered times, number of outliers removed, threshold used)
        """
        if len(times) == 0:
            return times, 0, float("inf")

        times_array = np.array(times)

        q1 = np.percentile(times_array, 25)
        q3 = np.percentile(times_array, 75)
        iqr = q3 - q1
        threshold = q3 + iqr_multiplier * iqr

        filtered = [t for t in times if t <= threshold]
        num_removed = len(times) - len(filtered)

        return filtered, num_removed, threshold

    def run_audit(self) -> dict:
        """
        Run complete audit and return results.

        :return: Dict containing all test data and metrics
        """
        miss_times, hit_times, miss_usage, hit_usage = self.interleaved_procedure()

        original_miss_count = len(miss_times)
        original_hit_count = len(hit_times)

        miss_times, miss_outliers, miss_threshold = self.filter_outliers_automatic(
            miss_times,
        )
        hit_times, hit_outliers, hit_threshold = self.filter_outliers_automatic(
            hit_times,
        )

        ks_statistic, p_value = self.compute_ks_test(hit_times, miss_times)

        metrics = self.compute_metrics(hit_times, miss_times)

        cache_token_analysis = self.analyze_cached_tokens(hit_usage, miss_usage)

        result = {
            "configuration": {
                "provider": self.provider_name,
                "scenario": self.scenario.value,
                "model": self.model,
                "prompt_length": self.prompt_length,
                "prefix_fraction": self.prefix_fraction,
                "num_samples": self.config["num_samples"],
                "num_victim_requests": self.config["num_victim_requests"],
                "delay_between_requests": self.delay_between_requests,
                "max_retries": self.max_retries,
            },
            "counts": {
                "expected_miss_count": self.config["num_samples"],
                "expected_hit_count": self.config["num_samples"],
                "actual_miss_count": original_miss_count,
                "actual_hit_count": original_hit_count,
                "miss_outliers_removed": miss_outliers,
                "hit_outliers_removed": hit_outliers,
                "final_miss_count": len(miss_times),
                "final_hit_count": len(hit_times),
            },
            "miss_times": miss_times,
            "hit_times": hit_times,
            "ks_statistic": float(ks_statistic),
            "p_value": float(p_value),
            "cache_detected_by_statistical_test": bool(p_value < 1e-8),
            "metrics": metrics,
            "cache_token_analysis": cache_token_analysis,
        }

        return result
