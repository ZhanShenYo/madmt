import ast
import json
import re

import json5
from openai import OpenAI


def is_japanese_str(input_string, thresh=0.4):
    """Return True if the string is mostly Japanese (kana + kanji ratio > thresh)."""
    if not input_string:
        return False
    input_string = re.sub(r'\d+', '', input_string)
    japanese_char_pattern = r'[\u3040-\u30FF\u4E00-\u9FFF]'
    num_jp = len(re.findall(japanese_char_pattern, input_string))
    ratio = num_jp / (len(input_string) + 1e-5)
    return ratio > thresh


def extract_clean_japanese_translation(text):
    """Pick a single clean Japanese sentence out of a model response.

    Drops romaji glosses in parentheses, splits alternative translations and
    keeps the longest candidate that is mostly Japanese. Returns "" if none.
    """
    text = re.sub(r'\([^)]*[a-zA-Zａ-ｚＡ-Ｚ]+\)', '', text)

    candidates = re.split(r'\n|\band\b|\bor\b|または|/|；|;|\d+\.', text)

    cleaned_candidates = []
    for cand in candidates:
        cand = cand.strip()
        if len(cand) < 2:
            continue
        if is_japanese_str(cand):
            cleaned_candidates.append(cand)

    if cleaned_candidates:
        return max(cleaned_candidates, key=len).strip()

    return ""


def repair_json_with_gpt(json_str, model):
    """Last-resort JSON repair: ask a small model to fix a malformed object.

    `model` comes from `repair_model` in configurations.yaml; the call is an
    OpenAI one, so it needs OPENAI_API_KEY regardless of the models used for
    translation.
    """
    client = OpenAI()
    prompt = f"""
You will be given a string that is intended to be a JSON object, but may contain formatting errors such as:
- unescaped double quotes inside values
- missing commas
- improper nesting
- mixed single and double quotes

Your task is to return a corrected JSON object that matches the user's intention, using only double quotes for keys and values.

Please return only the corrected JSON object, with no commentary, explanation, or markdown formatting.

Here is the original input:
{json_str}
"""
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that repairs broken JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
        )
        content = response.choices[0].message.content
        return json.loads(content)
    except Exception as e:
        print("[DEBUG] GPT-based JSON repair failed:", e)
        return None


def extract_json_objects_resilient(text, max_objects=10, repair_model=None):
    """Extract JSON objects from a raw LLM response.

    Tries, in order: strict JSON, JSON5, Python literal and quote fixing. When
    `repair_model` is set (from `repair_model` in configurations.yaml), a final
    model-based repair call is attempted; when it is empty, parsing simply
    fails instead of silently calling out to OpenAI.
    """
    json_objects = []
    stack = []
    start_idx = None

    text = re.sub(r"```json|```", "", text).strip()

    for i, char in enumerate(text):
        if char == '{':
            if not stack:
                start_idx = i
            stack.append('{')
        elif char == '}':
            if stack:
                stack.pop()
                if not stack and start_idx is not None:
                    json_str = text[start_idx:i+1]
                    parsed = False

                    try:
                        obj = json.loads(json_str)
                        json_objects.append(obj)
                        parsed = True
                    except Exception:
                        pass

                    if not parsed:
                        try:
                            obj = json5.loads(json_str)
                            json_objects.append(obj)
                            parsed = True
                        except Exception as e:
                            print("[DEBUG] JSON5 parsing failed:", e)

                    if not parsed:
                        try:
                            obj = ast.literal_eval(json_str)
                            json_objects.append(obj)
                            parsed = True
                        except Exception as e:
                            print("[DEBUG] ast.literal_eval parsing failed:", e)

                    if not parsed:
                        fallback_str = json_str
                        fallback_str = fallback_str.replace("'", '"')
                        fallback_str = re.sub(r'(?<!\\)"', '\\"', fallback_str)

                        try:
                            obj = json.loads(fallback_str)
                            json_objects.append(obj)
                            parsed = True
                        except Exception as final_e:
                            print("[DEBUG] JSON quote-replacement fallback failed:", final_e)

                    if not parsed and repair_model:
                        obj = repair_json_with_gpt(json_str, repair_model)
                        if obj:
                            json_objects.append(obj)
                            parsed = True

                    start_idx = None
                    if len(json_objects) >= max_objects:
                        break

    return json_objects
