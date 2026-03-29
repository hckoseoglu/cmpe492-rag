Reference turboquant: https://github.com/tonbistudio/turboquant-pytorch

================================================================================
TurboQuant Evaluation on Apple Silicon (MPS)
Dataset: MTEB SciFact
Embedding Model: Snowflake/snowflake-arctic-embed-s
================================================================================

1 - HOW TO RUN THE SCRIPT
================================================================================
The script evaluates the inner-product semantic search algorithm directly using PyTorch against TurboQuant's 'Asymmetric Attention' mathematical estimator. It operates on a given `--mode` representing varying levels of Quantization.

Requirements:
Ensure you run it with Python 3.9+ from this directory, using the provided variables.

  export USE_TF=0 USE_TORCH=1
  python benchmark_tq.py --mode [MODE]

Available MODES:
  --mode baseline (Executes standard FP32 vectors and baseline accuracy tests)
  --mode 4        (Executes 4-bit TurboQuant compression & retrieval)
  --mode 3        (Executes 3-bit TurboQuant compression & retrieval)
  --mode 2        (Executes 2-bit TurboQuant compression & retrieval)

The executed script will compile structural statistics inside `.json` elements placed within the directory (e.g., `results_baseline.json`) preserving the respective metric signatures (speed, memory footprint, ndcg@10). Visual evaluation processes can parse these configurations natively.

2 - ISSUES RECOGNIZED & APPLIED SOLUTIONS
================================================================================

ISSUE A: TensorFlow / PyTorch MPS Deadlock [mutex.cc : 452]
- Diagnosis: MTEB `datasets` integrations occasionally inject non-critical dependency loading instances under multiprocessing routines. The TensorFlow Abseil implementation internally locks the process when initializing its `mutex.cc` on a macOS architecture interacting concurrently with PyTorch's MPS stream blocks.
- Solution: Passed environment flags (`USE_TF=0 USE_TORCH=1`) enforcing HuggingFace native Torch behavior to clear the lock block permanently.

ISSUE B: OOM Device Locks During Native Batch Handling
- Diagnosis: Running a 560-Million parameter Transformer recursively onto 171,000 document texts heavily fragmented unified memory on Apple hardware, leading to silent hardware-level kernel hangs. 
- Solution: Applied a controlled validation constraint explicitly modifying native `model.encode(texts, batch_size=4)` memory streams, preventing GPU memory pools from flooding the system architectures.

3 - DATASET BRIEF (SCIFACT)
================================================================================
We utilized the MTEB SciFact dataset. It challenges models to verify scientific claims against a corpus of expert-written abstracts, functioning as a precise, smaller-scale semantic search challenge that evaluates extremely quickly without memory bottlenecks.
  - Documents (Corpus): 5,183 records
  - Queries: 1,109 target inquiries
  - Evaluation Relations (Test): 339 ground-truth validation pairings connecting inquiries to factual abstracts.

4 - EMBEDDING MODEL BRIEF
================================================================================
Model: `Snowflake/snowflake-arctic-embed-s` 
We swapped out our original dense retrieval arrays (`e5-large`) spanning 560M parameters as they consistently throttled hardware memory ceilings. Instead, we now leverage the highly-efficient open-source `snowflake-arctic-embed-s` model yielding exceptional performance footprint scales:
  - Total Parameters: 33 Million (0.033B)
  - Active Parameters: 22 Million (0.022B)
  - Dimensions: 384
  - Max Context Range: 512 Tokens
  - Retrieval Precision Mean: High baseline zero-shot accuracy across MTEB retrieval dimensions.
Despite being over 16x smaller than the previously structured e5 model, it computes its base inference logic cleanly across Apple Silicon memory lanes eliminating processing hangs while supporting precise baseline mathematical measurements for TurboQuant bit-compression testing.
