# Data provenance

Every corpus used by this project is recorded below; raw and processed data are gitignored
and never committed. Fill in the download date, version/commit, license and any filtering
applied at the moment a source is first fetched.

| Name | Source URL | Download date | Version/commit | License | Filtering applied |
|------|------------|---------------|----------------|---------|-------------------|
| Digital Corpus of Sanskrit (DCS) | https://github.com/OliverHellwig/sanskrit | not yet downloaded | | CC BY 4.0 | |
| SIGHUM | Krishna et al. 2017 release | not yet downloaded | | | |
| Hackathon | Krishnan et al. 2020 release | not yet downloaded | | | |
| DCS-2018 | Hellwig & Nehrdich 2018 release | not yet downloaded | | | |
| UoH corpus + SandhiKosh | https://sanskrit.uohyd.ac.in/Corpus/ | not yet downloaded | | | |
| Itihāsa | https://github.com/rahular/itihasa | 2026-09-03 | commit `37df077a80c83dbba1598afbdb079674b9bf6daa` | unspecified (repo has no LICENSE file; dataset described in Aralikatte et al., WAT 2021) | `data/{train,dev,test}.sn`/`.en` downloaded verbatim to `data/raw/itihasa/`; train 75,161 / dev 6,148 / test 11,721 aligned pairs; 0 empty pairs dropped in every split |
| Sāmayik | https://github.com/ayushbits/Saamayik (arXiv 2305.14004) | 2026-09-03 | commit `f87d54903e9f8540bda63efeb2c203e368195523` | unspecified (repo has no LICENSE file; dataset described in Aralikatte et al., LREC-COLING 2024) | `data/final_data/{train,dev,test}.sa`/`.en` and `data/mkb/mkb.sa`/`.en` (out-of-domain `test_ood`; sibling `mkb.txt` ignored, not an aligned pair) downloaded verbatim to `data/raw/samayik/`; train 43,493 / dev 2,416 / test 2,417 / test_ood 4,047 aligned pairs; 0 empty pairs dropped in every split |
| SAHAAYAK 2023 | arXiv 2307.00021 release | not yet downloaded | | | |
| FLORES-200 | https://dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz (official NLLB tarball; the HF ids `openlanguagedata/flores_plus`, `facebook/flores` and `Muennighoff/flores200` all failed, see `docs/decisions.md`) | 2026-09-03 | `flores200_dataset.tar.gz`, sha256 `b8b0b76783024b85797e5cc75064eb83fc5288b41e9654dabc7be6ae944011f6` | CC BY-SA 4.0 | `devtest` split only; `san_Deva`, `hin_Deva`, `eng_Latn` extracted verbatim (source script, no normalisation) to `data/raw/flores/devtest.jsonl`; 1012 aligned sentences |
| IN22-Gen | AI4Bharat | not yet downloaded | | | |
| ByT5-Sanskrit | HF `chronbmm/byt5-sanskrit` (verify exact id) | not yet downloaded | | | |
| DharmaBench | per-paper release | not yet downloaded | | | |
| IndicParam | per-paper release | not yet downloaded | | | |
