# Motion Data (Not Included)

The training clips are downloaded on demand from [lvhaidong/LAFAN1_Retargeting_Dataset](https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset), folder `g1/`.

The retargeted motions are identified by the upstream project as CC BY-NC-ND 4.0 (Ubisoft La Forge). They are not covered by this repository's MIT license and are intentionally excluded from GitHub releases and the project archive. Check the upstream dataset terms before use; commercial use and redistribution are not permitted by that license.

From the repository root, run:

```bash
uv run python tools/download_lafan1_g1.py
```

The downloader retrieves 40 CSV files and verifies each file against the Hugging Face tree response before storing it here.
