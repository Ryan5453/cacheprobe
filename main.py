import argparse
import json
from pathlib import Path

import yaml

from attacks import CachingAuditor, ScenarioType


def load_keys(keys_file: str = "keys.yaml") -> dict[str, dict]:
    """
    Load API keys from YAML configuration file.

    :param keys_file: Path to the YAML file containing API keys
    :return: Dictionary mapping provider names to API keys dict
    """
    keys_path = Path(keys_file)
    if not keys_path.exists():
        raise FileNotFoundError(f"Keys file '{keys_file}' not found.")
    with open(keys_path) as f:
        return yaml.safe_load(f)


def load_providers(config_file: str = "providers.json") -> dict[str, dict]:
    """
    Load provider configuration from JSON file.

    :param config_file: Path to the JSON file containing provider configs
    :return: Dictionary mapping provider names to configurations
    """
    config_path = Path(config_file)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file '{config_file}' not found.")

    with open(config_path) as f:
        return json.load(f)


def save_result(result: dict, provider: str, scenario: ScenarioType, results_dir: Path):
    """
    Save test result to JSON file.

    :param result: Dict containing audit results
    :param provider: The provider name
    :param scenario: The scenario type
    :param results_dir: Directory path for storing results
    """
    results_dir.mkdir(exist_ok=True)
    filename = f"{provider}_{scenario.value}.json"
    filepath = results_dir / filename

    with open(filepath, "w") as f:
        json.dump(result, f, indent=2)

    print(f"Results saved to: {filepath}")


def get_api_key(scenario: ScenarioType, provider: str, keys: dict[str, dict]) -> str:
    """
    Determine which API key to use based on scenario.

    :param scenario: The scenario type to test
    :param provider: The provider name
    :param keys: Dictionary of API keys for all providers
    :return: The appropriate API key to use
    """
    if scenario.value.startswith("openrouter"):
        return keys["openrouter"][
            "victim" if "same_account" in scenario.value else "attacker"
        ]
    else:
        return keys[provider][
            "victim" if "same_account" in scenario.value else "attacker"
        ]


def run_test(
    provider: str,
    scenario: ScenarioType,
    keys: dict[str, dict],
    providers: dict[str, dict],
    results_dir: Path,
) -> dict:
    """
    Run a single audit test.

    :param provider: The provider name to audit
    :param scenario: The scenario type to test
    :param keys: Dictionary of API keys for all providers
    :param providers: Dictionary of provider configurations
    :param results_dir: Directory path for storing results
    :return: Dict containing test results and metrics
    """
    api_key = get_api_key(scenario, provider, keys)
    config = providers[provider]
    auditor = CachingAuditor(provider, config, api_key, scenario)
    result = auditor.run_audit()
    save_result(result, provider, scenario, results_dir)
    return result


def print_test_results(result: dict):
    """
    Print formatted test results.

    :param result: Dict containing audit results
    """
    detected_caching = result["detected_caching"]
    print(f"\nTest completed! Caching detected: {detected_caching}")
    print(f"  - p-value: {result['p_value']}")
    print(f"  - KS statistic: {result['ks_statistic']}")

    metrics = result["metrics"]
    print(f"\nTiming Statistics:")
    print(f"  - Mean hit time: {metrics['mean_hit_time']}s")
    print(f"  - Mean miss time: {metrics['mean_miss_time']}s")
    print(f"  - Speedup: {metrics['mean_miss_time'] / metrics['mean_hit_time']}x")


def run_all_tests(keys: dict[str, dict], providers: dict[str, dict], results_dir: Path):
    """
    Run the complete test suite.

    :param keys: Dictionary of API keys for all providers
    :param providers: Dictionary of provider configurations
    :param results_dir: Directory path for storing results
    """
    test_plan = []
    for provider_name in providers.keys():
        test_plan.extend(
            [
                (provider_name, ScenarioType.DIRECT_SAME),
                (provider_name, ScenarioType.DIRECT_CROSS),
                (provider_name, ScenarioType.OR_DEFAULT_SAME),
                (provider_name, ScenarioType.OR_DEFAULT_CROSS),
            ]
        )

    for provider, scenario in test_plan:
        result = run_test(provider, scenario, keys, providers, results_dir)

        if scenario == ScenarioType.DIRECT_SAME and not result["detected_caching"]:
            print(f"WARNING: No caching detected for {provider}")
            print("Skipping remaining tests for this provider")
            break


def run_single_test(
    provider: str,
    scenario: str,
    keys: dict[str, dict],
    providers: dict[str, dict],
    results_dir: Path,
):
    """
    Run a single specific test.

    :param provider: Provider name as string
    :param scenario: Scenario name as string
    :param keys: Dictionary of API keys for all providers
    :param providers: Dictionary of provider configurations
    :param results_dir: Directory path for storing results
    """
    if provider not in providers:
        print(f"Error: Provider '{provider}' not found in config")
        print(f"\nAvailable providers: {list(providers.keys())}")
        return

    try:
        scenario_type = ScenarioType(scenario)
    except ValueError as e:
        print(f"Error: Invalid scenario - {e}")
        print(f"\nValid scenarios: {[s.value for s in ScenarioType]}")
        return

    result = run_test(provider, scenario_type, keys, providers, results_dir)
    print_test_results(result)


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
        help="Test mode: single (specific test) or all (full suite)",
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
    parser.add_argument(
        "--results-dir",
        default="results",
        help="Directory for storing results (default: results)",
    )

    args = parser.parse_args()

    try:
        keys = load_keys(args.keys_file)
        providers = load_providers(args.config)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return 1

    results_dir = Path(args.results_dir)

    if args.mode == "all":
        run_all_tests(keys, providers, results_dir)
    elif args.mode == "single":
        if not args.provider or not args.scenario:
            print("Error: --provider and --scenario required for single mode")
            return 1
        run_single_test(args.provider, args.scenario, keys, providers, results_dir)

    return 0


if __name__ == "__main__":
    exit(main())
