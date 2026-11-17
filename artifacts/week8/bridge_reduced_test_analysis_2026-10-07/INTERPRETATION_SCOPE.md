# Scope of the sealed BRIDGE reduced comparison

The hashed `RESULTS.md`, CSV files, and manifest in this directory are preserved as generated. Their numerical comparison is **between two methods built by this project**: the internal B2 matched pilot-feature baseline and the predefined `oeh_without_direct_oehb` ablation. BRIDGE supplies independently published MDPs; this project generated the offline pilot/training logs and the seven-class `E_l` labels.

This test asks whether the project's reduced difficulty-aware features transfer to the selected BRIDGE MDPs under a common forecast head. It does **not** compare OEH against an end-to-end method from another author, test the full 108-feature OEH, or establish state of the art. The reduced arm is worse for BC and better for CFQI and IQL on Brier and log loss; the budget-decision success rate does not improve.

The published TQ/SACo and ERI methods output dataset-characterization scores, not a seven-class distribution of learner-specific data requirements. An author-method comparison would require its original endpoint and access assumptions in a separate table, or a prespecified adaptation with a new common endpoint and an explicit `adapted` label. The image-based return predictor of Mateos-Melero et al. is a closer task analogue, but a return-to-`E_l` conversion would also be a new adaptation. No such external-author forecast arm was included in this sealed BRIDGE test.

The BRIDGE test outcomes are now open. Any new comparator selected or adapted after seeing them is exploratory on this panel. A confirmatory comparison needs its implementation, conversion rule, tuning data, task split, and analysis frozen before outcomes on a new untouched evaluation panel.
