# External Dataset & Test Fixture Locations (`DATA_LOCATIONS.md`)

> **Code-vs-Data Policy:** Raw FHI-aims real-space matrix exports (`rs_hamiltonian.out`, `rs_overlap.out`, `rs_soc_matrix.out/.bin`, `realspace_soc_matrix.out`) and converted DeepH `.h5` datasets are kept outside Git (`<25 MB` per-file cap enforced by `.githooks/pre-commit`).

| Dataset / Fixture | Host | Exact Path | Future Online Archive (Zenodo / Materials Cloud) |
| :--- | :--- | :--- | :--- |
| **Plain `rs_soc_matrix.out` Reference (`simple_mos2_calc`)** | Laptop (`adam`) | `/home/adam/work/FHI-aims/FHI-aims_rs_soc_hamiltonian_modifications/simple_mos2_calc` | *TBD — link DOI/URL once uploaded* |
| **Legacy Binary `realspace_soc_matrix` Reference (`MoS2_1x1_000`)** | Laptop (`adam`) | `/home/adam/work/ML-hamiltonians-workflow/sparsify_script_dev/sparsify_testing/example_data/fhi_DFT_data/MoS2_1x1_000_exx+0.0032_eyy+0.0078/soc_tight` | *TBD — link DOI/URL once uploaded* |
| **Production FHI-aims SOC `MoS2` Datasets** | `jhworkstation` | `/home/lacoxson/work/ML_Hamiltonians_workflow/01c_quoll_soc/datasets/` | *TBD — link DOI/URL once uploaded* |
