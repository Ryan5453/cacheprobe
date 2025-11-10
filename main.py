#!/usr/bin/env python3
"""
OpenRouter Timing Attack Auditor

Runs timing attack audits against LLM API providers to detect prompt caching vulnerabilities.
"""

import json
import yaml
import argparse
from copy import deepcopy
from pathlib import Path
from typing import Optional

from attacks import (
    CachingAuditor,
    ScenarioType,
)


class AuditRunner:
    def __init__(self, providers: dict[str, dict], results_dir: str = "results"):
        """
        Initialize the AuditRunner with provider configuration and results directory.

        :param providers: Dictionary of provider configurations
        :param results_dir: Directory path for storing audit results
        """
        self.providers = providers
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)

    @staticmethod
    def load_keys(keys_file: str = "keys.yaml") -> dict[str, dict]:
        """
        Load API keys from YAML configuration file.

        :param keys_file: Path to the YAML file containing API keys
        :return: Dictionary mapping provider names to API keys dict
        """
        keys_path = Path(keys_file)

        if not keys_path.exists():
            raise FileNotFoundError(
                f"Keys file '{keys_file}' not found. "
                f"Please copy 'keys.example.yaml' to '{keys_file}' and fill in your API keys."
            )

        with open(keys_path, "r") as f:
            return yaml.safe_load(f)

    @staticmethod
    def load_providers(config_file: str = "providers.json") -> dict[str, dict]:
        """
        Load provider configuration from JSON file.

        :param config_file: Path to the JSON file containing provider configs
        :return: Dictionary mapping provider names to configurations
        """
        config_path = Path(config_file)

        if not config_path.exists():
            raise FileNotFoundError(f"Config file '{config_file}' not found.")

        with open(config_path, "r") as f:
            return json.load(f)

    def run_test(
        self, provider: str, scenario: ScenarioType, keys: dict[str, dict]
    ) -> dict:
        """
        Run a single audit test.

        :param provider: The provider name to audit
        :param scenario: The scenario type to test
        :param keys: Dictionary of API keys for all providers
        :return: Dict containing test results and metrics
        """

        # Determine which API key to use
        if scenario.value.startswith("openrouter"):
            # OpenRouter scenarios
            if "same_account" in scenario.value:
                api_key = keys["openrouter"]["victim"]
            else:
                api_key = keys["openrouter"]["attacker"]
        else:
            # Direct API scenarios
            provider_keys = keys[provider]
            if "same_account" in scenario.value:
                api_key = provider_keys["victim"]
            else:
                api_key = provider_keys["attacker"]

        # Create auditor and run
        config = self.providers[provider]
        auditor = CachingAuditor(provider, config, api_key, scenario)
        result = auditor.run_audit()

        # Save result
        self.save_result(result, provider, scenario)

        return result

    def save_result(self, result: dict, provider: str, scenario: ScenarioType):
        """
        Save result to JSON file.

        :param result: Dict containing audit results
        :param provider: The provider name
        :param scenario: The scenario type
        """
        filename = f"{provider}_{scenario.value}.json"
        filepath = self.results_dir / filename

        with open(filepath, "w") as f:
            json.dump(result, f, indent=2)

        print(f"Results saved to: {filepath}")

    def load_result(self, provider: str, scenario: ScenarioType) -> Optional[dict]:
        """
        Load a saved result.

        :param provider: The provider name
        :param scenario: The scenario type
        :return: Dict if found, None otherwise
        """
        filename = f"{provider}_{scenario.value}.json"
        filepath = self.results_dir / filename

        if not filepath.exists():
            return None

        with open(filepath, "r") as f:
            return json.load(f)


def run_all_tests(keys: dict[str, dict], providers: dict[str, dict]):
    """
    Run the complete test suite.

    :param keys: Dictionary of API keys for all providers
    :param providers: Dictionary of provider configurations
    """
    runner = AuditRunner(providers)

    # Build test plan from configured providers
    test_plan = []
    for provider_name in providers.keys():
        test_plan.extend(
            [
                # Phase 1: Validate methodology
                (provider_name, ScenarioType.DIRECT_SAME),
                # Phase 2: Establish baselines
                (provider_name, ScenarioType.DIRECT_CROSS),
                # Phase 3: Main vulnerability tests
                (provider_name, ScenarioType.OR_DEFAULT_SAME),
                (provider_name, ScenarioType.OR_DEFAULT_CROSS),
            ]
        )

    results = []
    for provider, scenario in test_plan:
        result = runner.run_test(provider, scenario, keys)
        results.append(result)

        # Stop if we don't detect caching in validation phase
        detected_caching = result.get("detected_caching", result["p_value"] < 1e-8)
        if scenario == ScenarioType.DIRECT_SAME and not detected_caching:
            print(f"WARNING: No caching detected for {provider}")
            print("Skipping remaining tests for this provider")
            break

    print("\n" + "=" * 70)
    print("ALL TESTS COMPLETE")
    print("=" * 70)


def run_single_test(
    provider: str, scenario: str, keys: dict[str, dict], providers: dict[str, dict]
):
    """
    Run a single specific test.

    :param provider: Provider name as string
    :param scenario: Scenario name as string
    :param keys: Dictionary of API keys for all providers
    :param providers: Dictionary of provider configurations
    """
    runner = AuditRunner(providers)

    # Validate provider exists in config
    if provider not in providers:
        print(f"Error: Provider '{provider}' not found in config")
        print(f"\nAvailable providers: {list(providers.keys())}")
        return

    # Validate scenario
    try:
        scenario_type = ScenarioType(scenario)
    except ValueError as e:
        print(f"Error: Invalid scenario - {e}")
        print(f"\nValid scenarios: {[s.value for s in ScenarioType]}")
        return

    result = runner.run_test(provider, scenario_type, keys)
    print(f"\nTest completed!")

    # Show detection results
    detected_caching = result.get("detected_caching", result["p_value"] < 1e-8)
    print(f"Caching detected: {detected_caching}")

    if "detected_caching_timing" in result:
        print(f"  - Timing-based detection: {result['detected_caching_timing']}")
    if "detected_caching_provider" in result:
        print(f"  - Provider-reported detection: {result['detected_caching_provider']}")

    if "cache_stats" in result:
        stats = result["cache_stats"]
        print(f"\nCache Statistics:")
        print(
            f"  - Requests with cached tokens (hit): {stats['hit_requests_with_cache']}/{stats['total_hit_requests']}"
        )
        print(
            f"  - Requests with cached tokens (miss): {stats['miss_requests_with_cache']}/{stats['total_miss_requests']}"
        )
        if stats["avg_cached_tokens_hits"] > 0:
            print(
                f"  - Average cached tokens (hits): {stats['avg_cached_tokens_hits']:.0f}"
            )
            print(f"  - Cache hit rate: {stats['cache_hit_rate'] * 100:.1f}%")


def main():
    """
    Main entry point for the timing attack auditor.

    :return: Exit code (0 for success, 1 for error)
    """
    parser = argparse.ArgumentParser(
        description="Run timing attack audits on LLM API providers",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--mode",
        choices=["single", "all"],
        default="single",
        help="Test mode: single (specific test), or all (full suite)",
    )

    parser.add_argument(
        "--provider",
        help="Provider for single test mode (must match a provider in config file)",
    )

    parser.add_argument(
        "--scenario",
        choices=[s.value for s in ScenarioType],
        help="Scenario for single test mode",
    )

    parser.add_argument(
        "--keys-file",
        default="keys.yaml",
        help="Path to API keys YAML file (default: keys.yaml)",
    )

    parser.add_argument(
        "--config",
        default="providers.json",
        help="Path to provider configuration JSON file (default: providers.json)",
    )

    args = parser.parse_args()

    # Load configuration files
    try:
        keys = AuditRunner.load_keys(args.keys_file)
        providers = AuditRunner.load_providers(args.config)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return 1

    # Run appropriate mode
    if args.mode == "all":
        run_all_tests(keys, providers)
    if args.mode == "single":
        if not args.provider or not args.scenario:
            print("Error: --provider and --scenario required for single mode")
            return 1
        run_single_test(args.provider, args.scenario, keys, providers)

    return 0


if __name__ == "__main__":
    exit(main())
