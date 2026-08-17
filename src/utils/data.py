import json


def load_local_jsonl(data_path, start_num=0, end_num=1000):
    """Load sentence pairs from a JSONL file.

    Each line is a JSON object: {"source": "...", "target": "...", "context": "..."}
    """
    source_texts = []
    target_texts = []
    context_texts = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ex = json.loads(line)
            source_texts.append(ex["source"])
            target_texts.append(ex.get("target", ""))
            context_texts.append(ex.get("context", ""))

    return {
        "source_texts": source_texts[start_num:end_num],
        "target_texts": target_texts[start_num:end_num],
        "context_texts": context_texts[start_num:end_num],
    }
