# Week 6 E1/S1 final status — 27 September 2026

This document is the current compact index for the sealed Week 6 execution.
Earlier preflight requests, handovers and NO-GO reviews were removed after
their findings were closed. The final B2 review, E1 implementation review,
frozen protocols and raw local artifacts remain the audit authorities.

## E1 execution

- Run ID: `week6_oehv4e_confirmatory_v1_1`.
- Source commit: `725cbdb589d810b4c852b9776f01db0b29afe3dc`.
- Input manifest SHA-256:
  `1b93d48cde5a5ab69a7d3a365e01583807df2876812ee338bd62227c260ec2de`.
- Cohort: 120 base MDPs, comprising 84 stochastic primary MDPs and 36
  deterministic controls.
- Execution: 108,000 core fits and 24,000 external fits, complete with no
  retuning and no result filtering.

At x10 on the stochastic-primary cohort, OEH-v4E has mean normalized return
0.7881303911 and fold-best-fixed CFQI has 0.7833733471, a descriptive
difference of +0.0047570440. OEH-v4E success rate is 37.76%; CFQI success rate
is 41.19%. These endpoints answer different questions and are both retained.

## Corrected S1 analysis

The first S1 return adapter did not reproduce the registered salted inner-fold
assignment. Its checkpoint and verdict are retained locally but explicitly
invalidated. The correction preserves all 2,000 cluster multiplicities and all
three survival endpoints, and recomputes only `v4e` and `fold_best_fixed` using
the registered T1/E1 return procedure. E1 itself was not rerun.

For H1, the corrected observed effect is +0.0047570440 with basic 95% interval
[-0.0057765105, 0.0143224301]. The registered superiority gate does not pass:
the lower bound does not exceed the 0.005 practical margin and Holm does not
reject. The permitted claim is therefore descriptive leadership in mean
normalized return, not confirmed superiority over CFQI.

## Tracked evidence and local raw evidence

The repository tracks the compact seal, execution summary, diagnostics,
selector decisions, S1 summary, corrected verdict and invalidation record. The
309 MB raw execution tree remains unmodified in the local artifact directory
and is ignored by Git to avoid committing 132,640 per-job files.

Corrected artifact hashes:

- Bootstrap checkpoint:
  `bf6a0705db29c50e6e9b7fbca2a61eae684c3dab4517b54c266468442dd74a5a`.
- Bootstrap summary:
  `e7e52be7f4c2f6a6ca1d10e537d285f6e976d02f1b797b7fd143bb0c0d583487`.
- S1 verdict:
  `beb4604a169370384165df17e0517100eb31eb7522518c6e209f47b61e457c4d`.

The raw checkpoint is intentionally local; its SHA-256 remains in the tracked
summary and provenance records.
