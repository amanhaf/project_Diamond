# Evaluation and evidence limits

These results are historical project-report measurements, not independently reproduced benchmarks. The supplied files contain several revisions, including tracked editorial changes, incomplete result tables, and differing descriptions of the final identity pipeline.

## Runtime

The master report describes over five hours for sequential tagging of the 347-file corpus, 95.9 minutes after per-image concurrency, and approximately 35–50 minutes after also reducing the Pro model's reasoning budget. Attribute the final improvement to both concurrency and reasoning-budget control. Hardware, network conditions, API versions, and complete logs are not available here.

## Approximate identity

The threshold report describes comparison of all 60,031 unordered pairs among 347 renders. A threshold of at most 7 differing bits out of 256 corresponds to 97.27% bit agreement. This is hash agreement, not a calibrated probability that two designs are identical.

The report records four additional correct pair matches and zero observed incorrect pair matches at this threshold; the closest distinct pair was 8 bits apart. These are corpus-specific observations and do not guarantee zero false merges on new data or in transitive clusters. Proprietary sample names are omitted.

## Conflicting implementation status

The master report says tolerance logic existed in the cache but was not wired into the main driver and records rotation invariance as unfinished. The threshold report says the cache change was deployed and refers to a rotation fix. Without auditing source, commit chronology, and integration results against those reports, this case study treats the threshold as a reported recommendation and does not claim the final deployed behavior is verified.

Earlier implementation documents propose a combined SHA-256/perceptual signature. The master report supersedes that proposal with perceptual-hash-only UUID derivation.

## Unreported results

The master report retains placeholders for final classifier accuracy, per-class precision/recall/F1, a confusion matrix, and corrected per-image costs. No numerical claims for those items are made. Large catalog scale is a design target, not a measured deployment result.

## Reproduction requirements

Reproducing the original results would require authorized source code and design data, the trained model and label map, dependency versions, canonical rendering settings and namespace, evaluation group labels, hardware information, timing logs, and API usage records. These are not part of the supplied attachment set.
