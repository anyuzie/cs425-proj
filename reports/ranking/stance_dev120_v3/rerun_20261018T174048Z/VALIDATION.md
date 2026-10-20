# Rerun validation

All 38 targeted tests pass: 8 stance, 13 dense, 10 architecture and 7 configuration. Two outdated tests initially assumed the live dataset contained 60 profiles. The historical regression now uses frozen input snapshots; the CLI test checks the copied input record count. No scoring algorithm, dataset, historical predictions or metric definition changed.

All five conditions ran on the unchanged canonical 120-profile dataset. Aggregate and per-profile metrics reproduce the original stance_dev120_v3 runs exactly. Each fresh run is registered once in both evaluation histories; old rows are preserved. Runtime stance precision remains unverified without independent labels.
