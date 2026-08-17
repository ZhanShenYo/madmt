# DebatingAgent

A multi-agent debating system for machine translation evaluation and improvement. This project explores whether AI agents debating and collaborating can produce higher-quality translations than single-agent approaches.


## Quick start

### 1. Install

```bash
pip install -r requirements.txt
```

### 2. Configure API keys

```bash
cp .env.example .env
# then fill in the keys for the providers you plan to use
```

Only the providers named in `src/utils/configurations.yaml` need a key.

### 3. Run

```bash
# Baseline
python main.py -i ./examples/example.jsonl -o ./results/experiment -lp en-ja -s 0 -e 3 -mode zero

# Multi-agent methods
python main.py -i ./examples/example.jsonl -o ./results/experiment -lp en-ja -s 0 -e 3 -mode debate     -r 3
python main.py -i ./examples/example.jsonl -o ./results/experiment -lp en-ja -s 0 -e 3 -mode som        -r 3
python main.py -i ./examples/example.jsonl -o ./results/experiment -lp en-ja -s 0 -e 3 -mode reflection -r 3
python main.py -i ./examples/example.jsonl -o ./results/experiment -lp en-ja -s 0 -e 3 -mode decouple   -r 3
```

`scripts/run_example.sh` wraps the zero + SoM flow above.

## Command line arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `-i, --input-file` | Path to a JSONL data file | required |
| `-o, --output-dir` | Output directory | required |
| `-lp, --lang-pair` | Language pair (`en-ja`, `ja-en`) | required |
| `-mode, --mode` | `debate`, `reflection`, `som`, `decouple`, `zero` | `debate` |
| `-r, --round` | Number of rounds for the iterative methods | 3 |
| `-s, --start-num` | Start index of the data slice | 0 |
| `-e, --end-num` | End index of the data slice | 1 |
| `-max, --max-concurrency` | Maximum concurrent examples | 10 |

## Configuration

Hyperparameters and prompts are kept apart, and each setting has exactly one
source:

- `src/utils/configurations.yaml` — **hyperparameters only**
- `prompts/<mode>.json` — **all prompts** 

Placeholders such as `##source##`, `##src_lng##`, `##tgt_lng##`, `##context##`
and `##candidate_solutions##` are filled in at run time by
`src/utils/prompt.py`.

## Datasets

Obtain the data from its official
source and convert it to a JSONL file, one sentence pair per line:

```json
{"source": "The weather is nice today.", "target": "今日はいい天気です。", "context": ""}
```
The datasets used in our experiments are available from their official homepages:

- **WMT23** — https://machinetranslate.org/wmt23
- **KFTT** (Kyoto Free Translation Task) — https://www.phontron.com/kftt/
- **IWSLT 2017** (TED talks, WIT³ corpus) — https://huggingface.co/datasets/IWSLT/iwslt2017

## Output

```
results/experiment/
├── zero/
│   ├── {id}.json                    # per-example record
│   ├── {id}-config.json             # resolved prompts for that example
│   ├── run_log.json                 # success/failure per example
│   └── collected_translations.json  # base translations, reused by other modes
├── debate/
├── reflection/
├── som/
└── decouple/
```

## Citation

If you would like to cite the paper, here is a bibtex file:

```bibtex
@inproceedings{shen2026multi,
  title={Multi-Agent Debate for Machine Translation: A Case Study on English-Japanese Translation},
  author={Shen, Zhan and Naradowsky, Jason and Wang, Xiaotian and Miyao, Yusuke},
  booktitle={Proceedings of the 26th Annual Conference of the European Association for Machine Translation (Volume 1)},
  pages={205--230},
  year={2026}
}
```

## Acknowledgements

The agent and debate scaffolding is adapted from
[MAD: Multi-Agents-Debate](https://github.com/Skytliang/Multi-Agents-Debate)
(GPL-3.0).

## License

GPL-3.0 — see [LICENSE](LICENSE).
