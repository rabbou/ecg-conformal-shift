# The published code tables this study joins on

Three files, copied here unchanged so that the label table can be rebuilt from
the repository alone. Each is redistributable under the licence beside it, and
each is named where it is used.

| File | Source | Retrieved | Licence | SHA-256 |
|---|---|---|---|---|
| `dx_mapping_scored.csv` | `github.com/physionetchallenges/evaluation-2021`, branch `main` | 2026-09-07 | BSD 2-Clause, PhysioNet/Computing in Cardiology Challenges | `fad13ad9f7ca230e7e6392ac8a264cb7cd157879525129f964c5f708eabb41d0` |
| `AHA_SNOMED_mapping.csv` | `github.com/UTU-Health-Research/dl-ecg-classifier`, branch `main`, `data/` | 2026-09-07 | MIT, Tuija Leinonen 2024 | `23e0641aac859fd89ef8969402d844bb47bfb1caa6503bee451da359c5daeaae` |
| `ptbxlToSNOMED.csv` | PTB-XL+ 1.0.1, `labels/mapping/` on PhysioNet | 2026-09-07 | CC BY 4.0 | `63ae57a3a51387da39a6225a86ee2248b9cffc8d54215cbf177060ad59764d9f` |

`scripts/fetch_mappings.sh` re-downloads all three and checks these digests.

## What each one is for

`dx_mapping_scored.csv` is the Challenge-2021 scored-diagnosis table. It gives
the SNOMED CT code of every class the Challenge scored, the count of that class
in each of its eight partitions, and — in its `Notes` column — the pairs of
codes the Challenge scored as one diagnosis. Both the SNOMED codes the small
set is defined by and the counts the label table is checked against are read
from this file; none is typed in.

`AHA_SNOMED_mapping.csv` is Leinonen et al.'s AHA-to-SNOMED table, built with
two cardiologists (Comput Biol Med 2024, PMID 39427424). Shandong annotates in
the AHA scheme and is not a Challenge partition, so this is the bridge that puts
it in the same vocabulary as the other four corpora.

`ptbxlToSNOMED.csv` is PTB-XL+'s mapping from SCP-ECG acronyms to concepts. Its
`id1`..`id4` columns are **OMOP concept identifiers, not SNOMED CT codes**: the
file's own `apply_snomed_mapping.py` joins them against Athena's `CONCEPT.csv`,
which is not redistributable. So this file cannot by itself put PTB-XL's SCP
statements into the Challenge's SNOMED CT vocabulary. PTB-XL's SNOMED labels are
taken from the Challenge's own PTB-XL partition headers instead, and this file
is used for what it can do: name the clinical concept behind each SCP acronym,
so that the agreement between the two routes can be read and reported.
