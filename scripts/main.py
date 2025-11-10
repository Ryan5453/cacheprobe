#!/usr/bin/env python3
"""
OpenRouter Timing Attack Auditor

Runs timing attack audits against LLM API providers to detect prompt caching vulnerabilities.
"""

import json
import yaml
import argparse
from pathlib import Path
from typing import Optional

from attacks import (
    CachingAuditor,
    ProviderType,
    ScenarioType,
    APIKeys,
    AuditResult,
    PROVIDERS,
)


class AuditRunner:
    def __init__(self, results_dir: str = "results"):
        """
        Initialize the AuditRunner with a results directory.

        :param results_dir: Directory path for storing audit results
        """
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)

    @staticmethod
    def load_keys(keys_file: str = "keys.yaml") -> dict[str, APIKeys]:
        """
        Load API keys from YAML configuration file.

        :param keys_file: Path to the YAML file containing API keys
        :return: Dictionary mapping provider names to APIKeys objects
        """
        keys_path = Path(keys_file)

        if not keys_path.exists():
            raise FileNotFoundError(
                f"Keys file '{keys_file}' not found. "
                f"Please copy 'keys.example.yaml' to '{keys_file}' and fill in your API keys."
            )

        with open(keys_path, "r") as f:
            config = yaml.safe_load(f)

        # Convert to APIKeys objects
        return {
            provider: APIKeys(victim=keys["victim"], attacker=keys["attacker"])
            for provider, keys in config.items()
        }

    def run_test(
        self, provider: ProviderType, scenario: ScenarioType, keys: dict[str, APIKeys]
    ) -> AuditResult:
        """
        Run a single audit test.

        :param provider: The provider type to audit
        :param scenario: The scenario type to test
        :param keys: Dictionary of API keys for all providers
        :return: AuditResult containing test results and metrics
        """

        # Determine which API key to use
        if scenario.value.startswith("openrouter"):
            # OpenRouter scenarios
            if "same_account" in scenario.value:
                api_key = keys["openrouter"].victim
            else:
                api_key = keys["openrouter"].attacker
        else:
            # Direct API scenarios
            provider_keys = keys[provider.value]
            if "same_account" in scenario.value:
                api_key = provider_keys.victim
            else:
                api_key = provider_keys.attacker

        # Create auditor and run
        config = PROVIDERS[provider]
        auditor = CachingAuditor(provider, config, api_key, scenario)
        result = auditor.run_audit()

        # Save result
        self.save_result(result)

        return result

    def save_result(self, result: AuditResult):
        """
        Save result to JSON file.

        :param result: AuditResult object to save
        """
        filename = f"{result.provider.value}_{result.scenario.value}.json"
        filepath = self.results_dir / filename

        with open(filepath, "w") as f:
            f.write(result.model_dump_json(indent=2))

        print(f"Results saved to: {filepath}")

    def load_result(
        self, provider: ProviderType, scenario: ScenarioType
    ) -> Optional[AuditResult]:
        """
        Load a saved result.

        :param provider: The provider type
        :param scenario: The scenario type
        :return: AuditResult if found, None otherwise
        """
        filename = f"{provider.value}_{scenario.value}.json"
        filepath = self.results_dir / filename

        if not filepath.exists():
            return None

        with open(filepath, "r") as f:
            data = json.load(f)

        return AuditResult(**data)


def run_pilot(keys: dict[str, APIKeys]):
    """
    Run a quick pilot test with reduced samples.

    :param keys: Dictionary of API keys for all providers
    """
    runner = AuditRunner()

    print("Running pilot test with reduced samples...")
    pilot_config = PROVIDERS[ProviderType.GROQ].model_copy()
    pilot_config.num_samples = 50
    pilot_config.prompt_length = 500

    # Temporarily update config for pilot
    PROVIDERS[ProviderType.GROQ] = pilot_config

    result = runner.run_test(
        provider=ProviderType.GROQ, scenario=ScenarioType.DIRECT_SAME, keys=keys
    )

    print(f"\nPilot test completed!")
    print(f"Caching detected: {result.detected_caching}")


def run_all_tests(keys: dict[str, APIKeys]):
    """
    Run the complete test suite.

    :param keys: Dictionary of API keys for all providers
    """
    runner = AuditRunner()

    # Define test order
    test_plan = [
        # Phase 1: Validate methodology
        (ProviderType.GROQ, ScenarioType.DIRECT_SAME),
        (ProviderType.FIREWORKS, ScenarioType.DIRECT_SAME),
        # Phase 2: Establish baselines
        (ProviderType.GROQ, ScenarioType.DIRECT_CROSS),
        (ProviderType.FIREWORKS, ScenarioType.DIRECT_CROSS),
        (ProviderType.OPENAI, ScenarioType.DIRECT_SAME),
        (ProviderType.OPENAI, ScenarioType.DIRECT_CROSS),
        # Phase 3: Main vulnerability tests
        (ProviderType.GROQ, ScenarioType.OR_DEFAULT_SAME),
        (ProviderType.GROQ, ScenarioType.OR_DEFAULT_CROSS),
        (ProviderType.FIREWORKS, ScenarioType.OR_DEFAULT_SAME),
        (ProviderType.FIREWORKS, ScenarioType.OR_DEFAULT_CROSS),
        (ProviderType.OPENAI, ScenarioType.OR_DEFAULT_SAME),
        (ProviderType.OPENAI, ScenarioType.OR_DEFAULT_CROSS),
    ]

    results = []
    for provider, scenario in test_plan:
        result = runner.run_test(provider, scenario, keys)
        results.append(result)

        # Stop if we don't detect caching in validation phase
        if scenario == ScenarioType.DIRECT_SAME and not result.detected_caching:
            print(f"\n⚠️  WARNING: No caching detected for {provider.value}")
            print("Skipping remaining tests for this provider")
            break

    print("\n" + "=" * 70)
    print("ALL TESTS COMPLETE")
    print("=" * 70)


def run_single_test(
    provider: str, scenario: str, keys: dict[str, APIKeys]
):
    """
    Run a single specific test.

    :param provider: Provider name as string
    :param scenario: Scenario name as string
    :param keys: Dictionary of API keys for all providers
    """
    runner = AuditRunner()

    try:
        provider_type = ProviderType(provider)
        scenario_type = ScenarioType(scenario)
    except ValueError as e:
        print(f"Error: Invalid provider or scenario - {e}")
        print(f"\nValid providers: {[p.value for p in ProviderType]}")
        print(f"Valid scenarios: {[s.value for s in ScenarioType]}")
        return

    result = runner.run_test(provider_type, scenario_type, keys)
    print(f"\nTest completed!")
    print(f"Caching detected: {result.detected_caching}")


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
        choices=["pilot", "all", "single"],
        default="pilot",
        help="Test mode: pilot (quick test), all (full suite), or single (specific test)",
    )

    parser.add_argument(
        "--provider",
        choices=[p.value for p in ProviderType],
        help="Provider for single test mode",
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

    args = parser.parse_args()

    # Load API keys
    try:
        keys = AuditRunner.load_keys(args.keys_file)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return 1

    # Run appropriate mode
    if args.mode == "pilot":
        run_pilot(keys)
    elif args.mode == "all":
        run_all_tests(keys)
    elif args.mode == "single":
        if not args.provider or not args.scenario:
            print("Error: --provider and --scenario required for single mode")
            return 1
        run_single_test(args.provider, args.scenario, keys)

    return 0


if __name__ == "__main__":
    exit(main())

