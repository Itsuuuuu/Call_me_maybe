import argparse
import json
from pathlib import Path
import sys
from typing import Any, List

from llm_sdk import Small_LLM_Model
from src.decoder import ConstrainedDecoder
from src.models import FunctionCallResult, FunctionDefinition, PromptInput


def load_json(filepath: str) -> Any:
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Erreur : {filepath} not found.", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError:
        print(
            f"Erreur : {filepath} invalide JSON.",
            file=sys.stderr
        )
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="LLM Function Calling Engine"
    )
    parser.add_argument(
        "--functions_definition",
        type=str,
        default="data/input/functions_definition.json"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/input/function_calling_tests.json"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/output/function_calling_results.json"
    )

    args = parser.parse_args()

    raw_functions = load_json(args.functions_definition)
    functions = [FunctionDefinition(**f) for f in raw_functions]

    raw_prompts = load_json(args.input)
    prompts = [PromptInput(**p) for p in raw_prompts]

    print("LLM Download")
    try:
        model = Small_LLM_Model()
        decoder = ConstrainedDecoder(
            model, [f.model_dump() for f in functions]
        )
    except KeyboardInterrupt:
        print(
            "\nInterrupt during the modele's download", file=sys.stderr
        )
        sys.exit(130)
    except Exception as e:
        print(
            f"Critical error during modele's initialisation: {e}",
            file=sys.stderr
        )
        sys.exit(1)

    results: List[dict] = []

    try:
        for item in prompts:
            if not item.prompt or not item.prompt.strip():
                print(
                    "Found empty prompt or composed of spaces. "
                    "Prompt skypped, gone to the next."
                )
                continue
            print(f"\nPrompt treatment : '{item.prompt}'")

            raw_json_output = decoder.generate_function_call(item.prompt)
            print(f"Brut output: {raw_json_output}")

            try:
                parsed_json = json.loads(raw_json_output)
                parsed_json["prompt"] = item.prompt
                result = FunctionCallResult(**parsed_json)

                results.append(result.model_dump())
                print("JSON valid")

            except json.JSONDecodeError:
                print(
                    "Invalid JSON",
                    file=sys.stderr
                )
            except Exception as e:
                print(
                    f"Echec to validate pydantic file: {e}",
                    file=sys.stderr
                )

    except KeyboardInterrupt:
        print(
            "\nInterupt Ctrl+C used. "
            "Generation stopped.",
            file=sys.stderr
        )
        print("Generate results saved", file=sys.stderr)

    except Exception as e:
        print(
            f"Generating error: {e}",
            file=sys.stderr
        )
        print("Generate results saved", file=sys.stderr)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
        print(f"\nResuls saved in: {output_path}")
    except Exception as e:
        print(
            f"Error while writting: {e}",
            file=sys.stderr
        )
        sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nSTOP", file=sys.stderr)
        sys.exit(130)
