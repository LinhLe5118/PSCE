"""Resume frozen third-cohort preparation after dataset generation.

The archived evaluation package must be imported before pilot extraction when
the generated dataset already exists. This wrapper leaves the frozen runner,
forecast models, and decision rule unchanged.
"""
from scripts.analysis.run_unseen_validation import archive_imports


def main() -> None:
    archive_imports()
    from scripts.analysis.run_oeh_120_cross_capacity import prepare

    prepare()


if __name__ == "__main__":
    main()
