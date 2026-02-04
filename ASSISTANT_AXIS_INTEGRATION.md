# Integration with assistant-axis-main

This document explains how the activation analysis pipeline integrates with the `assistant-axis-main` codebase.

## Why Use assistant-axis?

The `assistant-axis-main` repository provides a mature, well-tested implementation of activation extraction with:

1. **Hook-based extraction**: More reliable than `output_hidden_states=True`
2. **Span mapping**: Accurate separation of prompt vs response tokens
3. **Batch processing**: Efficient extraction from multiple conversations
4. **Multi-layer support**: Extract from multiple layers in a single forward pass
5. **ProbingModel abstraction**: Unified interface for different model architectures

## What We Reuse

### Core Classes

From `assistant_axis.internals`:

1. **`ProbingModel`**: Loads models with standardized interface
   - Handles model initialization
   - Provides `get_layers()` for layer access
   - Supports tensor parallelism

2. **`ConversationEncoder`**: Formats conversations for models
   - Applies chat templates
   - Builds token spans for conversation turns
   - Handles special tokens correctly

3. **`ActivationExtractor`**: Extracts activations using forward hooks
   - Registers hooks on target layers
   - Captures activations during forward pass
   - Supports batch processing

4. **`SpanMapper`**: Maps activations to conversation spans
   - Identifies which tokens belong to each turn
   - Averages activations over turn tokens
   - Handles padding and attention masks

### Integration Points

Our pipeline integrates at these points:

```
experiments/activation_extraction.py
├── extract_activations_for_response()
│   ├── _extract_with_assistant_axis()  ← Uses assistant-axis internals
│   │   ├── ConversationEncoder()
│   │   ├── ActivationExtractor()
│   │   └── SpanMapper()
│   └── _extract_direct()               ← Fallback if unavailable
│
└── extract_activations_from_csv()
    └── Can load models with ProbingModel
```

## How It Works

### Example: Extracting Activations for a Response

```python
# With assistant-axis (preferred)
from assistant_axis.internals import ProbingModel
from experiments.activation_extraction import extract_activations_for_response

# Load model with ProbingModel
model = ProbingModel("unsloth/Qwen2.5-7B-Instruct")

# Extract activations
activations = extract_activations_for_response(
    model=model,
    tokenizer=model.tokenizer,
    question="What is 2+2?",
    response="The answer is 4.",
    layers_to_extract=[14, 20],
)
# Result: {14: array(...), 20: array(...)}
```

### What Happens Under the Hood

1. **Conversation Formatting**:
   ```python
   conversation = [
       {"role": "user", "content": question},
       {"role": "assistant", "content": response}
   ]
   ```

2. **Span Building**:
   - `ConversationEncoder` tokenizes and identifies turn boundaries
   - User tokens: [0, 50]
   - Assistant tokens: [50, 120]

3. **Activation Extraction**:
   - `ActivationExtractor` registers hooks on layers [14, 20]
   - Forward pass captures activations
   - Shape: (num_layers, batch_size, seq_len, hidden_size)

4. **Span Mapping**:
   - `SpanMapper` averages activations over assistant tokens only
   - Result: (num_layers, hidden_size)

5. **Output**:
   - Returns dict: {layer_idx: averaged_vector}

## Advantages Over Direct Extraction

### 1. Accurate Token Separation

**Direct approach** (original implementation):
```python
# Tokenize prompt alone to get length
prompt_tokens = tokenizer(prompt)
prompt_length = len(prompt_tokens)

# Then assume response starts at prompt_length
response_tokens = full_tokens[prompt_length:]
```

❌ Problem: Token counts may differ due to chat template formatting

**assistant-axis approach**:
```python
# Build explicit spans for each turn
spans = encoder.build_batch_turn_spans([conversation])
# Spans contain exact token indices: {role, start, end}
```

✅ Solution: Explicit span tracking ensures accuracy

### 2. Consistent with Existing Codebase

If you're already using `assistant-axis` for other experiments, this ensures:
- Same tokenization
- Same span detection
- Same activation extraction methodology
- Results are directly comparable

### 3. Handles Edge Cases

The assistant-axis implementation handles:
- Multi-turn conversations
- System prompts
- Special tokens
- Padding and truncation
- Different chat templates per model

## Fallback Mode

If `assistant-axis` is not available, the pipeline falls back to direct extraction:

```python
# Fallback: uses output_hidden_states=True
activations = _extract_direct(
    model, tokenizer, question, response, layers_to_extract
)
```

This ensures the pipeline works even without assistant-axis, but:
- ⚠️ Token separation is less accurate (length-based)
- ⚠️ May not handle edge cases as well
- ⚠️ Not guaranteed to match assistant-axis results

## Dependencies

The integration adds these dependencies (already in repo):

```python
from assistant_axis.internals import (
    ProbingModel,
    ConversationEncoder,
    ActivationExtractor,
    SpanMapper,
)
```

These are imported conditionally:
```python
try:
    from assistant_axis.internals import ...
    ASSISTANT_AXIS_AVAILABLE = True
except ImportError:
    ASSISTANT_AXIS_AVAILABLE = False
```

## Testing the Integration

### Quick Test

```python
from experiments.activation_extraction import ASSISTANT_AXIS_AVAILABLE

if ASSISTANT_AXIS_AVAILABLE:
    print("✓ assistant-axis integration active")
else:
    print("⚠ Using fallback extraction")
```

### Full Test

```bash
# Run test script
python << 'EOF'
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd() / "assistant-axis-main"))
sys.path.insert(0, str(Path.cwd() / "experiments"))

from assistant_axis.internals import ProbingModel
from activation_extraction import extract_activations_for_response

# Load small model for testing
model = ProbingModel("google/gemma-2-2b-it")  # or any model

# Extract
activations = extract_activations_for_response(
    model=model,
    tokenizer=model.tokenizer,
    question="Test question",
    response="Test response",
    layers_to_extract=[10],
)

print(f"✓ Extracted activations: layer 10 shape {activations[10].shape}")
EOF
```

## When to Use Which Implementation

### Use assistant-axis Integration (Recommended)

- ✅ When `assistant-axis-main` is available (default)
- ✅ For maximum accuracy
- ✅ For consistency with other experiments
- ✅ For complex conversations (multi-turn, system prompts)

### Use Fallback

- ⚠️ Only when assistant-axis unavailable
- ⚠️ For simple single-turn Q&A
- ⚠️ For quick prototyping

## Future Enhancements

Potential improvements to the integration:

1. **Batch processing**: Use `ActivationExtractor.batch_conversations()` for entire CSVs
2. **Multi-turn support**: Extend to handle multi-turn conversations from CSVs
3. **System prompts**: Add support for system prompts in CSV
4. **Streaming**: Process large CSV files without loading all into memory

## Summary

The integration with `assistant-axis-main`:

- ✅ Leverages existing, tested code
- ✅ Ensures accuracy in token separation
- ✅ Maintains consistency across experiments
- ✅ Handles edge cases robustly
- ✅ Falls back gracefully if unavailable

**Key Principle**: Reuse existing infrastructure when possible, maintain backward compatibility when necessary.
