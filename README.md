*This project has been created as part of the 42 curriculum by guifouqu.*

# Call Me Maybe - LLM Function Calling System

## Description
This project implements a precise function calling system that bridges natural language processing and structured computer execution. Using a small language model (Qwen3-0.6B), the system extracts intent and structured arguments from user prompts and maps them into deterministic, 100% parseable JSON outputs matching a predefined function schema.

Rather than relying purely on prompt engineering, which is highly unreliable with small models, this system utilizes constrained decoding to guide the generation token-by-token, guaranteeing absolute structural validity.

## Instructions

### Installation
This project manages dependencies using `uv`. To set up the virtual environment and install all mandatory packages (`pydantic`, `numpy`):

    uv sync

### Execution
The program runs as a module and processes input JSON files containing definitions and prompts.

Default run (using `data/input/` and writing to `data/output/`):

    uv run python -m src

Custom paths run:

    uv run python -m src --functions_definition custom_path/functions.json --input custom_path/tests.json --output custom_path/results.json

### Code Quality & Linting
To check code quality against Flake8 and Mypy typing rules:

    make lint

For strict type checking:

    make lint-strict

---

## Algorithm Explanation
The core logic relies on **Constrained Decoding** rather than standard greedy sampling. 

1. **Logit Interception**: At each token generation step, the model produces raw logits representing the probability distribution over the whole vocabulary.
2. **Schema Masking**: Based on the current state of the generated JSON string, a mask is computed. We map the allowed characters or structure (e.g., forcing quotes, colons, or matching specific argument keys from `functions_definition.json`) to valid token IDs using the vocabulary JSON file.
3. **Logit Modification**: Tokens that would violate the syntax rules or the expected Pydantic schema have their logits heavily penalized to negative infinity (`-inf`), leaving only structurally correct choices.
4. **Deterministic Token Selection**: The next token is sampled exclusively from the remaining valid options, ensuring the final output is 100% parseable JSON.

## Design Decisions
* **Pydantic Validation**: All functions and argument payloads are strictly modeled using Pydantic classes to guarantee input/output schema validation and clean object parsing.
* **State Machine for JSON**: Implemented a lightweight syntax tracking mechanism to check whether the model is inside a key, a value, or generating structural syntax.
* **Graceful Error Handling**: Input operations, missing files, or malformed payloads are wrapped in specific try-except blocks using context managers to avoid unexpected crashes and resource leakage.

## Performance Analysis
* **Accuracy**: Achieves a 96% accuracy rate on correct function selection and 93% accuracy on argument extraction, passing the required 90% threshold.
* **Reliability**: Generates 100% valid JSON parseable output across multiple runs, eliminating structural hallucinations entirely.
* **Speed**: Processes the entire test prompt suite in approximately 1 minute and 20 seconds, well within the 5-minute hardware constraint.

## Challenges Faced
* *Token Overlap Bias*: Standard tokenizers include leading spaces or split simple words into unexpected subword units. Resolving how token fragments fit into the strict schema boundaries required building a robust token lookahead validation loop.
* *Handling Nested Architectures*: Ensuring that typed values (like numbers or booleans) were strictly forced without allowing text or prose outside the JSON formatting structure.

## Testing Strategy
The implementation was extensively validated against multiple scenarios:
1. **Edge Cases**: Empty strings, large integers, negative floating numbers, and special characters inside text arguments.
2. **Robustness Tests**: Running with missing input definition files or heavily malformed JSON input files to ensure proper fallback error logs are provided instead of crashing.
3. **Ambiguity**: Evaluated prompts that don't match any function schema to verify how the pipeline handles unmapped logic gracefully.

## Resources
* HuggingFace & Qwen Tokenizer documentation.
* Guidance and Constrained Decoding design articles.

### AI Usage Disclosure
Artificial Intelligence (such as Large Language Models) was used during development for the following tasks:
* Drafting linter configuration workflows and setting up Makefile rules.
* Generating edge-case text inputs for unit testing.
* Refining specific parts of the token lookahead filtering logic.