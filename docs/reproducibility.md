# Reproducibility

The repository contains a minimal source workflow and a single direct dependency. Run the smoke suite with Python, compileall, and pytest as shown in the README. Keep input tables, fitted weights, predictions, and generated outputs outside version control.

For a real study, archive the exact source revision, dependency lock, configuration, split manifest, preprocessing state, feature definitions, seeds, and result artifacts. Do not describe a run as reproducible when its source dependency closure or input lineage is incomplete.

The historical Phase4 artifacts are not recreated by this repository. No historical metric is included here.
