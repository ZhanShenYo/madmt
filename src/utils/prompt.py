import json
import os

import yaml

PROMPTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "prompts",
)


def prompts_path_for(mode):
    """Path of the prompt file backing a given mode."""
    return os.path.join(PROMPTS_DIR, f"{mode}.json")


class PromptBuilder:
    """Fills the per-example prompt file used by one run.

    Prompts come from prompts/<mode>.json; configurations.yaml supplies the
    hyperparameters, so a prompt has exactly one source.
    """

    def __init__(self, config_path=None, base_path=None, id=None):
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        self.base_path = base_path
        with open(base_path, "r", encoding="utf-8") as f:
            self.base_prompts = json.load(f)
        self.id = id

    def _save(self):
        with open(self.base_path, 'w', encoding='utf-8') as file:
            json.dump(self.base_prompts, file, ensure_ascii=False, indent=4)

    def introduce_base_translation(self, zero_path):
        """Reuse the zero-shot output as the base translation for this example."""
        with open(zero_path, 'r', encoding='utf-8') as f:
            zero_file = json.load(f)
        base_translation = zero_file[str(self.config["base_model"])][str(self.id)]

        if base_translation is not None:
            self.base_prompts["base_translation"] = base_translation

        self._save()
