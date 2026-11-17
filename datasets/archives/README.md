# Sealed E1 raw dataset archive

`week6_e1_oehv4e_confirmatory_v1_1.tar.gz` is a snapshot of the local E1 dataset directory. It contains 6,600 `buffer-*.npz` files plus `manifest.json`, `data_only_index.json`, and `generation_access_log.json`. The archive is stored with Git LFS; the two large JSON indexes are also available at their original paths through Git LFS.

After cloning and running `git lfs pull` from the repository root, restore the buffers with:

```powershell
tar -xzf datasets/archives/week6_e1_oehv4e_confirmatory_v1_1.tar.gz -C datasets
```

The SHA-256 of the archive is `da0601ab14d336b33dd308470712f3bd423a6bd1cfc53cb2e1fffca1355e67f0`. The E1 generation access log records SHA-256 `1b93d48cde5a5ab69a7d3a365e01583807df2876812ee338bd62227c260ec2de` for `manifest.json` and `60e5603feaff054cd5a155fcaeb55f812df85eadbcdf39a0c5419647bc29bc07` for `data_only_index.json`.

The saved A2 features allow refitting the seven-cohort forecaster without unpacking this archive. Other cohorts' raw buffers are not included in this archive.
