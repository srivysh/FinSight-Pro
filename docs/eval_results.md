# Retrieval Configuration Evaluation

## Objective

Evaluate multiple retrieval configurations and select the best one based on empirical evaluation rather than intuition.

## Configurations Tested

| Configuration | Pass Rate | Avg Keyword | Avg Confidence | Avg Latency | Citation Rate |
|---------------|----------:|------------:|---------------:|------------:|--------------:|
| Vector (k=4) | 75% | 0.75 | 0.68 | 9.09 s | 100% |
| Vector (k=8) | **95%** | **0.97** | 0.74 | 40.88 s | **100%** |
| Hybrid (k=4) | 65% | 0.64 | 0.77 | 9.50 s | 100% |
| Hybrid (k=6) | 85% | 0.72 | **0.84** | 16.92 s | 100% |

## Final Decision

The final production configuration is **Vector Search with k=8**.

### Why this configuration?

The retrieval strategy was selected based on objective evaluation using a custom benchmark of 20 financial question-answer pairs covering five SEC 10-K filings.

Although Hybrid Search achieved the highest average confidence (0.84), Vector Search with **k=8** produced the highest overall pass rate (95%) while maintaining a perfect citation rate (100%).

Since answer correctness is the primary objective of the system, Vector Search with k=8 was selected as the production retrieval configuration.