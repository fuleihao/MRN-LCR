## Setup

```bash

pip install -r requirements.txt
cd transformers && pip install -e . && cd ..
```

Follow [PL-Marker](https://github.com/thunlp/PL-Marker) to prepare ACE04 / ACE05 / SciERC.

Download SciBERT / BERT / ALBERT-xxlarge into `bert_models/`:

```text
bert_models/scibert_scivocab_uncased
bert_models/bert-base-uncased
bert_models/albert-xxlarge-v1
```

Optional: use [HGERE](https://github.com/yanzhh/HGERE) NER predictions (`$HGERE_ROOT`) for better end-to-end scores.

## Quickstart (SciERC)

```bash
bash shell/scierc/scibert.sh   # SciERC
bash shell/ace05/bert.sh       # ACE05 + BERT
bash shell/ace05/albert.sh     # ACE05 + ALBERT
bash shell/ace04/bert.sh       # ACE04 5-fold
bash shell/ace04/albert.sh 
```

Before running `shell/*.sh`, set `$GPU_ID` and replace `--test_file` / `--output_dir` with your NER path (`$HGERE_ROOT/...` or local `*ner_models/...`) and output path.

## Notes

- Install `transformers` from this repo (`pip install -e .`), not PyPI.
- Paper numbers average 5 seeds; single-seed results may vary.
- Best SciERC uses HGERE NER; switch `--test_file` accordingly.
