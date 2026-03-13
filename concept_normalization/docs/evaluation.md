# Evaluation Plan & Failure Analysis

## Gold Standard Test Set

Create hand-labeled query pairs per submodule. Each test case has an input query and expected output. A domain expert in exercise science should review all cases.

| Submodule | Min Cases | Breakdown |
|-----------|-----------|-----------|
| Abbreviation Expansion | 50 | 30 with abbreviations, 10 without (should pass through unchanged), 10 edge cases (ambiguous abbreviations, mixed case) |
| Coordination Resolution | 40 | 15 compound queries, 10 single-concept compounds that should NOT decompose, 10 comparisons (vs/versus), 5 list queries |
| Term Variation Handling | 60 | 30 lay-term queries, 15 already-canonical queries, 15 ambiguous/edge cases |

## Metrics

| Metric | What It Measures | Target |
|--------|-----------------|--------|
| Exact Match (EM) | % of outputs matching gold standard exactly | >85% abbreviations, >75% coordination, >70% term variation |
| Semantic Similarity | Cosine similarity between output and gold standard embeddings (same model as RAG retriever) | >0.95 average |
| Retrieval Improvement | Recall@k with vs without normalization on queries with known relevant docs | >10% relative improvement in Recall@5 |

## Evaluation Protocol

1. **Unit tests per submodule**: Run each submodule independently against its gold standard slice. Measure EM and semantic similarity.
2. **Integration tests**: Run the full pipeline end-to-end on compound test cases exercising all three submodules. Verify transformation log correctness at each step.
3. **Retrieval A/B test**: Run RAG retriever on 100 queries with and without normalization. Measure Recall@5 and MRR.
4. **Regression monitoring**: After vocabulary updates or prompt changes, re-run full test suite and compare to baseline.

---

## Failure Modes

### Abbreviation Expansion

| Failure Mode | Example | Mitigation |
|-------------|---------|------------|
| Ambiguous abbreviation | "PT" = physical therapy or personal training | Check vocabulary for context clues. If ambiguous, preserve original and flag both candidates in log. |
| Not in vocabulary | User uses novel/niche abbreviation | Preserve unchanged. Log as "unknown abbreviation". |
| Over-expansion | "SET" (noun) misidentified as abbreviation | Only expand exact vocabulary abbreviation matches. Case-sensitive matching for short forms. |
| JSON parse failure | Model returns malformed JSON | Retry once with explicit JSON reminder. If still fails, pass query unchanged and log error. |

### Coordination Resolution

| Failure Mode | Example | Mitigation |
|-------------|---------|------------|
| False decomposition of compound concept | "strength and conditioning" split into two | Check vocabulary for compound terms before decomposing. If term exists as unit, do not split. |
| Missed coordination | Comma-separated list not decomposed | Detect comma lists and "and"/"or" conjunctions explicitly in prompt. |
| Incomplete context distribution | "Benefits of X and Y for Z" loses "for Z" on one sub-query | Prompt explicitly requires distributing shared context to all sub-queries. |
| Over-decomposition | "rest and recovery" split when closely related | Check vocabulary. If both map to similar concepts, preserve as single query. |

### Term Variation Handling

| Failure Mode | Example | Mitigation |
|-------------|---------|------------|
| Over-normalization | "toning" mapped to generic "exercise" | Require specific vocabulary match. Prompt emphasizes preserving domain-specific meaning. |
| Wrong disambiguation | "bad knees" mapped to wrong condition | Flag low-confidence. Include alternatives in log. Preserve original as fallback. |
| Hallucinated mapping | LLM invents normalization not in vocabulary | Post-processing validation: check every normalized term against vocabulary. Revert if not found. |
| Loss of query intent | Normalization loses specificity | Normalize only domain terms. Preserve structural and intent-bearing words unchanged. |

## Uncertainty Handling Protocol

1. **Confidence threshold**: Any normalization with confidence below 0.7 is flagged as uncertain.
2. **Dual-query strategy**: For uncertain normalizations, output BOTH original and normalized versions as separate retriever queries.
3. **Fallback to original**: If LLM call fails entirely (timeout, malformed response, rate limit), original query passes through unchanged with error flag.
4. **No silent failures**: Every decision, including decisions NOT to normalize, is recorded in the transformation log.

## Fallback Mechanisms

| Scenario | Behavior |
|----------|----------|
| LLM returns invalid JSON | Retry once with explicit JSON reminder appended. If second attempt fails, pass query unchanged. |
| LLM timeout / rate limit | Pass query unchanged with error flag. Alert monitoring. |
| Vocabulary file missing or corrupted | Raise startup error. Do not serve queries without vocabulary grounding. |
| Empty or nonsensical query | Pass through unchanged. Log as "no normalization needed". |
| Normalization degrades retrieval | Feature flag to bypass individual submodules. A/B test to confirm. |

## Production Monitoring

- **Latency per submodule**: Track P50/P95/P99 for each LLM call. Alert if total pipeline exceeds 3 seconds.
- **Normalization rate**: % of queries modified per submodule. Sudden change = prompt drift or vocabulary issue.
- **Confidence distribution**: Monitor score distribution. Shift toward lower confidence = vocabulary needs updating.
- **Retrieval quality**: Periodically run gold standard retrieval test. Recall@5 below baseline = investigate.
- **JSON parse failure rate**: Track per submodule. Above 2% = prompt revision needed.
