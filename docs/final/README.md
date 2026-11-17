# Final documentation

These pages describe the final state of the project. Everything else under `docs/` is the record of how it got there,
kept unchanged under the evidence rules.

Read in this order:

1. [METHOD.md](METHOD.md): the problem, the label, the baseline B2 and the six blocks of OEH with their frozen settings.
2. [DATA.md](DATA.md): the sources, behavior policies and collection steps, and how the data are split between
   meta-training, development and the final tests.
3. [FINAL_TESTS.md](FINAL_TESTS.md): protocols, seals, amendments and registered results of the three final tests, and
   what the evidence supports. Generated from the artifacts by `python -m scripts.reproduce.final_tests_doc`.
4. [HISTORY.md](HISTORY.md): how the final method was reached, with every earlier sealed test and where it is recorded.

Related:

- [Root README](../../README.md): how to reproduce the results on a new machine, the repository map, and the names used
  in the code.
- [artifacts/README.md](../../artifacts/README.md): which artifact folders the final pipeline reads.
- The paper sources are in [`paper/`](../../paper/); its final PDF will be added there after every claim has been
  checked against its artifact (checkpoint H6).
