import argparse
import sys
from pathlib import Path
from src.utils import data
from src.methods import Debate
from src.methods import Reflection
from src.methods import SoM
from src.methods import Decouple
from src.methods import Zero_Shot
sys.path.append(str(Path(__file__).parent))

import os
import json
import dotenv
import yaml

import asyncio
from functools import partial
from datetime import datetime
from tqdm.asyncio import tqdm_asyncio
import traceback
from src.utils.prompt import PromptBuilder, prompts_path_for

CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src/utils/configurations.yaml")


def generation_settings():
    """The `generation` block from configurations.yaml."""
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        settings = yaml.safe_load(f)
    return settings.get("generation", {})


def sleep_time_for(mode):
    """Per-method pause between API calls, from configurations.yaml."""
    return generation_settings().get("sleep_time", {}).get(mode, 0)


def temperature_setting():
    """Sampling temperature, from configurations.yaml."""
    return generation_settings().get("temperature", 0)


def run_debate_task(id, source, reference, context, config, save_file_dir, src_full, tgt_full, api_keys, round):
    try:
        prompts_path = f"{save_file_dir}/{id}-config.json"

        config['source'] = source
        config['reference'] = reference
        config['context'] = context
        config['src_lng'] = src_full
        config['tgt_lng'] = tgt_full


        with open(prompts_path, 'w', encoding='utf-8') as file:
            json.dump(config, file, ensure_ascii=False, indent=4)


        current_path = os.path.abspath(__file__)
        tem_path = current_path.rsplit("/", 1)[0]
        config_path = os.path.join(tem_path, "src/utils/configurations.yaml")
        builder = PromptBuilder(config_path=config_path, base_path=prompts_path,id=id)

        save_file_parent_dir = os.path.dirname(save_file_dir)
        zero_path = os.path.join(save_file_parent_dir, "zero", "collected_translations.json")
        if os.path.exists(zero_path):
            builder.introduce_base_translation(zero_path)

        debate = Debate(
            save_file_dir=save_file_dir,
            api_keys=api_keys,
            prompts_path=prompts_path,
            temperature=temperature_setting(),
            round=round,
            sleep_time=sleep_time_for("debate"),
            config_path=config_path
        )

        debate.run()
        debate.save_file_to_json(id)

        return {"id": id, "status": "success"}

    except Exception as e:
        return {"id": id, "status": "error", "error": str(e), "traceback": traceback.format_exc()}

def run_reflection_task(id, source, reference, context, config, save_file_dir, src_full, tgt_full, api_keys, round):
    try:
        prompts_path = f"{save_file_dir}/{id}-config.json"

        config['source'] = source
        config['reference'] = reference
        config['context'] = context
        config['src_lng'] = src_full
        config['tgt_lng'] = tgt_full

        with open(prompts_path, 'w', encoding='utf-8') as file:
            json.dump(config, file, ensure_ascii=False, indent=4)

        current_path = os.path.abspath(__file__)
        tem_path = current_path.rsplit("/", 1)[0]
        config_path = os.path.join(tem_path, "src/utils/configurations.yaml")
        builder = PromptBuilder(config_path=config_path, base_path=prompts_path,id=id)
        save_file_parent_dir = os.path.dirname(save_file_dir)
        zero_path = os.path.join(save_file_parent_dir, "zero", "collected_translations.json")
        if os.path.exists(zero_path):
            builder.introduce_base_translation(zero_path)

        reflect = Reflection(
            save_file_dir=save_file_dir,
            api_keys=api_keys,
            prompts_path=prompts_path,
            temperature=temperature_setting(),
            round=round,
            sleep_time=sleep_time_for("reflection"),
            config_path=config_path
        )

        reflect.run()
        reflect.save_file_to_json(id)

        return {"id": id, "status": "success"}

    except Exception as e:
        return {"id": id, "status": "error", "error": str(e), "traceback": traceback.format_exc()}

def run_som_task(id, source, reference, context, config, save_file_dir, src_full, tgt_full, api_keys, round):
    try:
        prompts_path = f"{save_file_dir}/{id}-config.json"

        config['source'] = source
        config['reference'] = reference
        config['context'] = context
        config['src_lng'] = src_full
        config['tgt_lng'] = tgt_full

        with open(prompts_path, 'w', encoding='utf-8') as file:
            json.dump(config, file, ensure_ascii=False, indent=4)

        current_path = os.path.abspath(__file__)
        tem_path = current_path.rsplit("/", 1)[0]
        config_path = os.path.join(tem_path, "src/utils/configurations.yaml")
        builder = PromptBuilder(config_path=config_path, base_path=prompts_path,id=id)
        save_file_parent_dir = os.path.dirname(save_file_dir)
        zero_path = os.path.join(save_file_parent_dir, "zero", "collected_translations.json")
        if os.path.exists(zero_path):
            builder.introduce_base_translation(zero_path)

        som = SoM(
            save_file_dir=save_file_dir,
            api_keys=api_keys,
            prompts_path=prompts_path,
            temperature=temperature_setting(),
            round=round,
            sleep_time=sleep_time_for("som"),
            config_path=config_path
        )

        som.run()
        som.save_file_to_json(id)

        return {"id": id, "status": "success"}

    except Exception as e:
        return {"id": id, "status": "error", "error": str(e), "traceback": traceback.format_exc()}

def run_decouple_task(id, source, reference, context, config, save_file_dir, src_full, tgt_full, api_keys, round):
    try:
        prompts_path = f"{save_file_dir}/{id}-config.json"

        config['source'] = source
        config['reference'] = reference
        config['context'] = context
        config['src_lng'] = src_full
        config['tgt_lng'] = tgt_full

        with open(prompts_path, 'w', encoding='utf-8') as file:
            json.dump(config, file, ensure_ascii=False, indent=4)

        current_path = os.path.abspath(__file__)
        tem_path = current_path.rsplit("/", 1)[0]
        config_path = os.path.join(tem_path, "src/utils/configurations.yaml")
        builder = PromptBuilder(config_path=config_path, base_path=prompts_path,id=id)
        save_file_parent_dir = os.path.dirname(save_file_dir)
        zero_path = os.path.join(save_file_parent_dir, "zero", "collected_translations.json")
        if os.path.exists(zero_path):
            builder.introduce_base_translation(zero_path)

        decouple = Decouple(
            save_file_dir=save_file_dir,
            api_keys=api_keys,
            prompts_path=prompts_path,
            temperature=temperature_setting(),
            round=round,
            sleep_time=sleep_time_for("decouple"),
            config_path=config_path
        )

        decouple.run()
        decouple.save_file_to_json(id)

        return {"id": id, "status": "success"}

    except Exception as e:
        return {"id": id, "status": "error", "error": str(e), "traceback": traceback.format_exc()}

def run_zero_task(id, source, reference, context, config, save_file_dir, src_full, tgt_full, api_keys):
    try:
        prompts_path = f"{save_file_dir}/{id}-config.json"

        config['source'] = source
        config['reference'] = reference
        config['context'] = context
        config['src_lng'] = src_full
        config['tgt_lng'] = tgt_full

        with open(prompts_path, 'w', encoding='utf-8') as file:
            json.dump(config, file, ensure_ascii=False, indent=4)

        current_path = os.path.abspath(__file__)
        tem_path = current_path.rsplit("/", 1)[0]
        config_path = os.path.join(tem_path, "src/utils/configurations.yaml")

        zero = Zero_Shot(
            save_file_dir=save_file_dir,
            api_keys=api_keys,
            prompts_path=prompts_path,
            temperature=temperature_setting(),
            sleep_time=sleep_time_for("zero"),
            config_path=config_path
        )

        zero.save_file_to_json(id)

        return {"id": id, "status": "success"}

    except Exception as e:
        return {"id": id, "status": "error", "error": str(e), "traceback": traceback.format_exc()}

def parse_args():
    parser = argparse.ArgumentParser("", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-i", "--input-file", type=str, required=True, help="Path to a JSONL data file (see README)")
    parser.add_argument("-o", "--output-dir", type=str, required=True, help="Output file dir")
    parser.add_argument("-lp", "--lang-pair", type=str, required=True, help="Language pair")
    parser.add_argument("-r", "--round", type=int, default=3, help="Number of rounds")
    parser.add_argument("-e", "--end-num", type=int, default=1, help="End data number")
    parser.add_argument("-s", "--start-num", type=int, default=0, help="Start data number")
    parser.add_argument("-mode", "--mode", type=str, default="debate", help="debate or reflection or both")
    parser.add_argument("-max", "--max-concurrency", type=int, default=10, help="max concurrency")
    return parser.parse_args()

def get_api_key():
    """Read API keys from the environment (loaded from .env)."""
    api_keys = {
        "openai": os.getenv("OPENAI_API_KEY"),
        "anthropic": os.getenv("ANTHROPIC_API_KEY"),
        "deepseek": os.getenv("DEEPSEEK_API_KEY"),
        "gemini": os.getenv("GEMINI_API_KEY"),
        "azure": os.getenv("AZURE_KEY"),
        "azure_endpoint": os.getenv("AZURE_ENDPOINT")
    }

    env_names = {
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "deepseek": "DEEPSEEK_API_KEY",
        "gemini": "GEMINI_API_KEY",
        "azure": "AZURE_OPENAI_KEY",
        "azure_endpoint": "AZURE_OPENAI_ENDPOINT"
    }
    for key, env_name in env_names.items():
        if api_keys[key]:
            os.environ[env_name] = api_keys[key]
    return api_keys

def report(results, log_path):
    n_success = sum(1 for r in results if r['status'] == 'success')
    n_error = sum(1 for r in results if r['status'] == 'error')
    print(f"\nFinished {len(results)} tasks: {n_success} succeeded, {n_error} failed.")
    print(f"Details saved to {log_path}")




if __name__ == "__main__":
    args = parse_args()

    # apikey
    dotenv.load_dotenv()
    api_keys = get_api_key()


    # data
    dataset = data.load_local_jsonl(args.input_file, args.start_num, args.end_num)
    source_texts = dataset["source_texts"]
    target_texts = dataset["target_texts"]
    context_texts = dataset["context_texts"]

    max_concurrency = args.max_concurrency

    try:
        if args.mode == 'debate':
            if args.lang_pair == "en-ja":
                src_lng, tgt_lng = "en", "ja"
                src_full = "English"
                tgt_full = "Japanese"

            elif args.lang_pair == "ja-en":
                src_lng, tgt_lng = "ja", "en"
                src_full = "Japanese"
                tgt_full = "English"

            config = json.load(open(prompts_path_for("debate"), "r"))

            # output dir
            if not os.path.exists(args.output_dir):
                    os.mkdir(args.output_dir)
            save_file_dir = os.path.join(args.output_dir, "debate")
            if not os.path.exists(save_file_dir):
                    os.mkdir(save_file_dir)

            async def limited_run(fn, sem):
                async with sem:
                    return await asyncio.to_thread(fn)

            async def debate_fun():
                sem = asyncio.Semaphore(max_concurrency)
                tasks = []


                for id, (source, reference, context) in enumerate(zip(source_texts, target_texts, context_texts)):
                    task_fn = partial(run_debate_task, id, source, reference, context, config.copy(), save_file_dir, src_full, tgt_full, api_keys, args.round)
                    task = limited_run(task_fn, sem)
                    tasks.append(task)

                results = await tqdm_asyncio.gather(*tasks)

                log_path = os.path.join(save_file_dir, "run_log.json")
                with open(log_path, "w", encoding='utf-8') as f:
                    json.dump(results, f, ensure_ascii=False, indent=4)

                report(results, log_path)

            asyncio.run(debate_fun())

        elif args.mode == 'reflection':
            if args.lang_pair == "en-ja":
                src_lng, tgt_lng = "en", "ja"
                src_full = "English"
                tgt_full = "Japanese"

            elif args.lang_pair == "ja-en":
                src_lng, tgt_lng = "ja", "en"
                src_full = "Japanese"
                tgt_full = "English"
            config = json.load(open(prompts_path_for("reflection"), "r"))

            if not os.path.exists(args.output_dir):
                os.mkdir(args.output_dir)
            save_file_dir = os.path.join(args.output_dir, "reflection")
            if not os.path.exists(save_file_dir):
                os.mkdir(save_file_dir)

            async def limited_run(fn, sem):
                async with sem:
                    return await asyncio.to_thread(fn)

            async def reflection_fun():
                sem = asyncio.Semaphore(max_concurrency)
                tasks = []

                for id, (source, reference, context) in enumerate(zip(source_texts, target_texts, context_texts)):
                    task_fn = partial(run_reflection_task, id, source, reference, context, config.copy(), save_file_dir, src_full, tgt_full, api_keys, args.round)
                    task = limited_run(task_fn, sem)
                    tasks.append(task)

                results = await tqdm_asyncio.gather(*tasks)

                log_path = os.path.join(save_file_dir, "run_log.json")
                with open(log_path, "w", encoding='utf-8') as f:
                    json.dump(results, f, ensure_ascii=False, indent=4)

                report(results, log_path)

            asyncio.run(reflection_fun())

        elif args.mode == 'som':
            if args.lang_pair == "en-ja":
                src_lng, tgt_lng = "en", "ja"
                src_full = "English"
                tgt_full = "Japanese"

            elif args.lang_pair == "ja-en":
                src_lng, tgt_lng = "ja", "en"
                src_full = "Japanese"
                tgt_full = "English"

            elif args.lang_pair == "en-zh":
                src_lng, tgt_lng = "en", "zh"
                src_full = "English"
                tgt_full = "Chinese"

            config = json.load(open(prompts_path_for("som"), "r"))

            if not os.path.exists(args.output_dir):
                os.mkdir(args.output_dir)
            save_file_dir = os.path.join(args.output_dir, "som")
            if not os.path.exists(save_file_dir):
                os.mkdir(save_file_dir)

            async def limited_run(fn, sem):
                async with sem:
                    return await asyncio.to_thread(fn)

            async def som_fun():
                sem = asyncio.Semaphore(max_concurrency)
                tasks = []

                for id, (source, reference, context) in enumerate(zip(source_texts, target_texts, context_texts)):
                    task_fn = partial(run_som_task, id, source, reference, context, config.copy(), save_file_dir, src_full, tgt_full, api_keys, args.round)
                    task = limited_run(task_fn, sem)
                    tasks.append(task)

                results = await tqdm_asyncio.gather(*tasks)

                log_path = os.path.join(save_file_dir, "run_log.json")
                with open(log_path, "w", encoding='utf-8') as f:
                    json.dump(results, f, ensure_ascii=False, indent=4)

                report(results, log_path)

            asyncio.run(som_fun())

        elif args.mode == 'zero':
            if args.lang_pair == "en-ja":
                src_lng, tgt_lng = "en", "ja"
                src_full = "English"
                tgt_full = "Japanese"

            elif args.lang_pair == "ja-en":
                src_lng, tgt_lng = "ja", "en"
                src_full = "Japanese"
                tgt_full = "English"

            config = json.load(open(prompts_path_for("zero"), "r"))

            if not os.path.exists(args.output_dir):
                os.mkdir(args.output_dir)
            save_file_dir = os.path.join(args.output_dir, "zero")
            if not os.path.exists(save_file_dir):
                os.mkdir(save_file_dir)

            async def limited_run(fn, sem):
                async with sem:
                    return await asyncio.to_thread(fn)

            async def zero_fun():
                sem = asyncio.Semaphore(max_concurrency)
                tasks = []

                for id, (source, reference, context) in enumerate(zip(source_texts, target_texts, context_texts)):
                    task_fn = partial(run_zero_task, id, source, reference, context, config.copy(), save_file_dir, src_full, tgt_full, api_keys)
                    task = limited_run(task_fn, sem)
                    tasks.append(task)

                results = await tqdm_asyncio.gather(*tasks)

                log_path = os.path.join(save_file_dir, "run_log.json")
                with open(log_path, "w", encoding='utf-8') as f:
                    json.dump(results, f, ensure_ascii=False, indent=4)

                report(results, log_path)

            asyncio.run(zero_fun())

            # Collect every base translation into a single file, used as the
            # base translation by the other modes.
            folder_path = save_file_dir

            results = {}

            for filename in sorted(os.listdir(folder_path)):
                if filename.endswith('.json') and filename[:-5].isdigit():
                    file_path = os.path.join(folder_path, filename)
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        model_name = data.get("base_name", "unknown_model")
                        index = int(filename[:-5])
                        translation = data.get("base_translation", "")

                        if model_name not in results:
                            results[model_name] = {}
                        results[model_name][index] = translation

            collected_output_path = os.path.join(folder_path, 'collected_translations.json')
            with open(collected_output_path, 'w', encoding='utf-8') as f_out:
                json.dump(results, f_out, ensure_ascii=False, indent=2)

        elif args.mode == 'decouple':
            if args.lang_pair == "en-ja":
                src_lng, tgt_lng = "en", "ja"
                src_full = "English"
                tgt_full = "Japanese"

            elif args.lang_pair == "ja-en":
                src_lng, tgt_lng = "ja", "en"
                src_full = "Japanese"
                tgt_full = "English"

            config = json.load(open(prompts_path_for("decouple"), "r"))

            # output dir
            if not os.path.exists(args.output_dir):
                    os.mkdir(args.output_dir)
            save_file_dir = os.path.join(args.output_dir, "decouple")
            if not os.path.exists(save_file_dir):
                    os.mkdir(save_file_dir)

            async def limited_run(fn, sem):
                async with sem:
                    return await asyncio.to_thread(fn)

            async def decouple_fun():
                sem = asyncio.Semaphore(max_concurrency)
                tasks = []


                for id, (source, reference, context) in enumerate(zip(source_texts, target_texts, context_texts)):
                    task_fn = partial(run_decouple_task, id, source, reference, context, config.copy(), save_file_dir, src_full, tgt_full, api_keys, args.round)
                    task = limited_run(task_fn, sem)
                    tasks.append(task)

                results = await tqdm_asyncio.gather(*tasks)

                log_path = os.path.join(save_file_dir, "run_log.json")
                with open(log_path, "w", encoding='utf-8') as f:
                    json.dump(results, f, ensure_ascii=False, indent=4)

                report(results, log_path)

            asyncio.run(decouple_fun())

    except Exception as e:
        print(f"Error occurred: {str(e)}")
        sys.exit(1)
