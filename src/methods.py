import os
import json
import re
import yaml
from src.utils.agent import Agent
from datetime import datetime
from src.utils.text_utils import extract_clean_japanese_translation
from src.utils.text_utils import extract_json_objects_resilient


class Player(Agent):
    def __init__(self, model_name: str, name: str, temperature:float, api_keys: dict, sleep_time: float) -> None:
        """Create a player in the debate

        Args:
            model_name(str): model name
            name (str): name of this player
            temperature (float): higher values make the output more random, while lower values make it more focused and deterministic
            api_key (str): As the parameter name suggests
            sleep_time (float): sleep because of rate limits
        """
        super(Player, self).__init__(model_name, name, temperature, sleep_time)

class Debate:
    def __init__(self,
            temperature: float=0, 
            save_file_dir: str=None,
            api_keys: dict=None,
            prompts_path: str=None,
            round: int=3,
            sleep_time: float=0,
            config_path: str=None
        ) -> None:
        """Create a debate

        Args:
            model_name (str): model name
            temperature (float): higher values make the output more random, while lower values make it more focused and deterministic
            save_file_dir (str): dir path to json file
            api_key (str): As the parameter name suggests
            prompts_path (str): prompts path (json file)
            round (int): Rounds of Debate
            sleep_time (float): sleep because of rate limits
        """
        with open (config_path, 'r') as f:
            config = yaml.safe_load(f)
        self.NAME_DICT = config['NAME_DICT']
        self.repair_model = config.get('repair_model')
        self.model_name = config['debate_model_name']
        self.base_name = config['debate_base_model']
        self.temperature = temperature
        self.save_file_dir = save_file_dir
        self.api_keys = api_keys
        self.round = round
        self.sleep_time = sleep_time

        # init save file
        now = datetime.now()
        current_time = now.strftime("%Y-%m-%d_%H:%M:%S")
        self.save_file = {
            'start_time': current_time,
            'end_time': '',
            'base_name': self.base_name,
            'proponent': '',
            'opponent': '',
            'moderator': '',
            'temperature': temperature,
            "src_lng": "",
            "tgt_lng": "",
            'source': '',
            'reference': '',
            "debate_translation": '',
            "bleu_scores": [],
            "translations": [],
            "rounds" : [],
            'players': {},
        }
        prompts = json.load(open(prompts_path))
        self.save_file.update(prompts)
        self.init_prompt()

        if self.save_file['base_translation'] == "":
            print("No zero-shot base translation available; generating one.")
            self.create_base()
        self.save_file["translations"].append(self.save_file['base_translation'])

        # creat&init agents
        self.creat_agents()
        self.init_agents()

    def init_prompt(self):
        if self.save_file["context"] != "" and not self.save_file["context"].startswith("This is a translation context, please refer to it: "):
            self.save_file["context"] = "This is the translation context, please refer to it: " + self.save_file["context"] + " "

        def prompt_replace(key):
            self.save_file[key] = (
                self.save_file[key]
                .replace("##src_lng##", self.save_file["src_lng"])
                .replace("##tgt_lng##", self.save_file["tgt_lng"])
                .replace("##source##", self.save_file["source"])
                .replace("##base_translation##", self.save_file["base_translation"])
                .replace("##context##", self.save_file["context"] if self.save_file["context"] != "" else "")
            )

        prompt_replace("base_prompt")
        prompt_replace("player_meta_prompt")
        prompt_replace("moderator_meta_prompt")
        prompt_replace("affirmative_prompt") 

    def create_base(self):
        agent = Player(model_name=self.base_name, name='Baseline', temperature=self.temperature, api_keys=self.api_keys, sleep_time=self.sleep_time)
        agent.add_event(self.save_file['base_prompt'])
        base_translation = agent.ask()

        if self.save_file['tgt_lng'] == 'Japanese':
            base_translation = extract_clean_japanese_translation(base_translation)
        
        agent.add_memory(base_translation)
        self.save_file['base_translation'] = base_translation
        self.save_file['affirmative_prompt'] = self.save_file['affirmative_prompt'].replace("##base_translation##", base_translation)
        self.save_file['players'][agent.name] = agent.memory_lst
         
    def creat_agents(self):
        self.players = [
            Player(model_name=self.NAME_DICT[name], name=name, temperature=self.temperature, api_keys=self.api_keys, sleep_time=self.sleep_time) for name in self.NAME_DICT
        ]
        self.affirmative = self.players[0]
        self.negative = self.players[1]
        self.moderator = self.players[2]
        self.save_file['proponent'] = self.affirmative.model_name
        self.save_file['opponent'] = self.negative.model_name
        self.save_file['moderator'] = self.moderator.model_name

    def init_agents(self):
        # start: set meta prompt
        self.affirmative.set_meta_prompt(self.save_file['player_meta_prompt'])
        self.negative.set_meta_prompt(self.save_file['player_meta_prompt'])
        self.moderator.set_meta_prompt(self.save_file['moderator_meta_prompt'])
        
        # start: first round debate, state opinions
        self.affirmative.add_event(self.save_file['affirmative_prompt'])
        self.aff_ans = self.affirmative.ask()
        self.affirmative.add_memory(self.aff_ans)

        self.negative.add_event(self.save_file['negative_prompt'].replace('##aff_ans##', self.aff_ans))
        self.neg_ans = self.negative.ask()
        self.negative.add_memory(self.neg_ans)

        self.moderator.add_event(self.save_file['moderator_prompt'].replace('##aff_ans##', self.aff_ans).replace('##neg_ans##', self.neg_ans).replace('##round##', 'first'))
        self.mod_ans = self.moderator.ask()
        self.moderator.add_memory(self.mod_ans)
        cleaned_mod_ans = re.sub(r"```json|```", "", self.mod_ans).strip()
        self.mod_ans = json.loads(cleaned_mod_ans)


        if "debate_translation" in self.mod_ans and self.mod_ans["debate_translation"]:
            self.mod_ans["debate_translation"] = self.mod_ans["debate_translation"].strip("「」")
            if self.save_file["tgt_lng"] == 'Japanese':
                self.mod_ans["debate_translation"] = extract_clean_japanese_translation(self.mod_ans["debate_translation"])
            self.save_file["translations"].append(self.mod_ans["debate_translation"])

        round_data = {
            "round": 0,
            "affirmative": self.aff_ans,
            "negative": self.neg_ans,
            "moderator": self.mod_ans,
        }
        self.save_file["rounds"].append(round_data)

    def round_dct(self, num: int):
        dct = {
            1: 'first', 2: 'second', 3: 'third', 4: 'fourth', 5: 'fifth', 6: 'sixth', 7: 'seventh', 8: 'eighth', 9: 'ninth', 10: 'tenth'
        }
        return dct[num]
            
    def save_file_to_json(self, id):
        now = datetime.now()
        current_time = now.strftime("%Y-%m-%d_%H:%M:%S")
        save_file_path = os.path.join(self.save_file_dir, f"{id}.json")
        
        self.save_file['end_time'] = current_time
        json_str = json.dumps(self.save_file, ensure_ascii=False, indent=4)
        with open(save_file_path, 'w') as f:
            f.write(json_str)

    def broadcast(self, msg: str):
        """Broadcast a message to all players. 
        Typical use is for the host to announce public information

        Args:
            msg (str): the message
        """
        # print(msg)
        for player in self.players:
            player.add_event(msg)

    def speak(self, speaker: str, msg: str):
        """The speaker broadcast a message to all other players. 

        Args:
            speaker (str): name of the speaker
            msg (str): the message
        """
        if not msg.startswith(f"{speaker}: "):
            msg = f"{speaker}: {msg}"
        # print(msg)
        for player in self.players:
            if player.name != speaker:
                player.add_event(msg)

    def ask_and_speak(self, player: Player):
        ans = player.ask()
        player.add_memory(ans)
        self.speak(player.name, ans)

    def run(self):
        for round in range(self.round-1):
            # print(f"===== Debate Round-{round+1} =====\n")
            self.affirmative.add_event(self.save_file['debate_prompt'].replace('##oppo_ans##', self.neg_ans).replace('##pre_mod##', self.mod_ans["debate_translation"]))
            self.aff_ans = self.affirmative.ask()
            self.affirmative.add_memory(self.aff_ans)

            self.negative.add_event(self.save_file['debate_prompt'].replace('##oppo_ans##', self.aff_ans).replace('##pre_mod##', self.mod_ans["debate_translation"]))
            self.neg_ans = self.negative.ask()
            self.negative.add_memory(self.neg_ans)


            self.moderator.add_event(self.save_file['moderator_prompt'].replace('##aff_ans##', self.aff_ans).replace('##neg_ans##', self.neg_ans).replace('##round##', self.round_dct(round+2)))
            self.mod_ans = self.moderator.ask()
            self.moderator.add_memory(self.mod_ans)
            cleaned_mod_ans = re.sub(r"```json|```", "", self.mod_ans).strip()
            self.mod_ans = json.loads(cleaned_mod_ans)


            if "debate_translation" in self.mod_ans and self.mod_ans["debate_translation"]:
                self.mod_ans["debate_translation"] = self.mod_ans["debate_translation"].strip("「」")
                if self.save_file["tgt_lng"] == 'Japanese':
                    self.mod_ans["debate_translation"] = extract_clean_japanese_translation(self.mod_ans["debate_translation"])
                self.save_file["translations"].append(self.mod_ans["debate_translation"])

            round_data = {
                "round": round + 1,
                "affirmative": self.aff_ans,
                "negative": self.neg_ans,
                "moderator": self.mod_ans,
            }
            self.save_file["rounds"].append(round_data)


        if self.mod_ans["debate_translation"] != '':
            self.save_file["debate_translation"] = self.mod_ans["debate_translation"]


        for player in self.players:
            self.save_file['players'][player.name] = player.memory_lst

class Reflection:
    def __init__(self,
            temperature: float=0,
            save_file_dir: str=None,
            api_keys: dict=None,
            prompts_path: str=None,
            round: int=3,
            sleep_time: float=0,
            config_path: str=None
        ) -> None:

        with open (config_path, 'r') as f:
            config = yaml.safe_load(f)        
        self.repair_model = config.get('repair_model')
        self.model_name = config['reflction_model_name']
        self.base_name = config['reflction_base_model']
        self.temperature = temperature
        self.save_file_dir = save_file_dir
        self.api_keys = api_keys
        self.round = round
        self.sleep_time = sleep_time

        # init save file
        now = datetime.now()
        current_time = now.strftime("%Y-%m-%d_%H:%M:%S")
        self.save_file = {
            'start_time': current_time,
            'end_time': '',
            'model_name': self.base_name,
            'temperature': temperature,
            "src_lng": "",
            "tgt_lng": "",
            'source': '',
            'reference': '',
            'base_translation': '',
            "base_translation_bleuscore": '',
            'labels': [],
            "reflection_translation": '',
            "bleu_scores": [],
            "translations": [],
            "rounds" : [],
        }
        prompts = json.load(open(prompts_path))
        self.save_file.update(prompts)
        self.init_prompt()

        # creat reflection agents
        self.creat_agents()

    def init_prompt(self):
        if self.save_file["context"] != "" and not self.save_file["context"].startswith("This is a translation context, please refer to it: "):
            self.save_file["context"] = "This is a translation context, please refer to it: " + self.save_file["context"] + " "

        def prompt_replace(key):
            self.save_file[key] = (
                self.save_file[key]
                .replace("##src_lng##", self.save_file["src_lng"])
                .replace("##tgt_lng##", self.save_file["tgt_lng"])
                .replace("##source##", self.save_file["source"])
                .replace("##base_translation##", self.save_file["base_translation"])
                .replace("##context##", self.save_file["context"] if self.save_file["context"] != "" else "")
            )

        prompt_replace("base_prompt")
        prompt_replace("reflection_prompt")

    def creat_agents(self):
        self.agent = Player(model_name=self.model_name, name='Reflection', temperature=self.temperature, api_keys=self.api_keys, sleep_time=self.sleep_time)

    def save_file_to_json(self, id):
        now = datetime.now()
        current_time = now.strftime("%Y-%m-%d_%H:%M:%S")
        save_file_path = os.path.join(self.save_file_dir, f"{id}.json")
        
        self.save_file['end_time'] = current_time
        json_str = json.dumps(self.save_file, ensure_ascii=False, indent=4)
        with open(save_file_path, 'w') as f:
            f.write(json_str)
        

    def run(self):
        if self.save_file['base_translation'] == "":
            print("No external base translation was loaded; generating one.")
            self.agent.add_event(self.save_file['base_prompt'])
            base_translation = self.agent.ask()
            if self.save_file["tgt_lng"] == 'Japanese':
                base_translation = extract_clean_japanese_translation(base_translation)
            self.agent.add_memory(base_translation)
            self.save_file['base_translation'] = base_translation
        pre_translation = self.save_file['base_translation']
        self.save_file['translations'].append(self.save_file['base_translation'])

        round_data = {
            "round": 0,
            "base_translation": self.save_file['base_translation'],
        }
        self.save_file["rounds"].append(round_data)

        for round_idx in range(self.round):
            # eval
            self.agent.add_event(self.save_file['eval_prompt'])
            eval = self.agent.ask()
            self.agent.add_memory(eval)
            cleaned_mod_eval = re.sub(r"```json|```", "", eval).strip()
            label = json.loads(cleaned_mod_eval)
            # reflection
            self.save_file['labels'].append(label['label'])
            pre_label = label['label']

            self.agent.add_event(self.save_file['reflection_prompt'].replace('##pre_tran##', pre_translation).replace('##label##', pre_label))
            reflection_translation = self.agent.ask()

            if self.save_file["tgt_lng"] == 'Japanese':
                reflection_translation = extract_clean_japanese_translation(reflection_translation)
            pre_translation = reflection_translation
            self.agent.add_memory(reflection_translation)
            self.save_file['translations'].append(reflection_translation)


            round_data = {
                "round": round_idx + 1,
                "base_translation": self.save_file['base_translation'],
                "label": label['label'],
                "reflection_translation": reflection_translation,
            }
            self.save_file["rounds"].append(round_data)

        self.save_file['memory'] = self.agent.memory_lst
        
class SoM:
    def __init__(self,
            temperature: float=0, 
            save_file_dir: str=None,
            api_keys: dict=None,
            prompts_path: str=None,
            round: int=3,
            sleep_time: float=0,
            config_path: str=None
        ) -> None:

        with open (config_path, 'r') as f:
            config = yaml.safe_load(f)
        self.repair_model = config.get('repair_model')
        self.Society = config['Society']
        self.moderator_name = config['som_moderator_model']
        self.base_name = config['som_base_model']
        self.temperature = temperature
        self.save_file_dir = save_file_dir
        self.api_keys = api_keys
        self.round = round
        self.sleep_time = sleep_time

        # init save file
        now = datetime.now()
        current_time = now.strftime("%Y-%m-%d_%H:%M:%S")
        self.save_file = {
            'start_time': current_time,
            'end_time': '',
            'base_name': self.base_name,
            'temperature': temperature,
            "src_lng": "",
            "tgt_lng": "",
            'source': '',
            'reference': '',
            'base_translation': '',
            "debate_translation": '',
            'players': {},
            "rounds" : [],
            "bleu_scores": [],
            "translations": [],
        }
        prompts = json.load(open(prompts_path))
        self.save_file.update(prompts)
        self.init_prompt()

        if self.save_file['base_translation'] == "":
            self.create_base()
        self.save_file["translations"].append(self.save_file['base_translation'])

        # creat&init agents
        self.creat_agents()
        self.init_agents()

    def create_base(self):
        agent = Player(model_name=self.base_name, name='Baseline', temperature=self.temperature, api_keys=self.api_keys, sleep_time=self.sleep_time)
        agent.add_event(self.save_file['base_prompt'])
        base_translation = agent.ask()
        agent.add_memory(base_translation)

        if self.save_file['tgt_lng'] == 'Japanese':
            base_translation = extract_clean_japanese_translation(base_translation)
        
        self.save_file['base_translation'] = base_translation
        self.save_file['players'][agent.name] = agent.memory_lst

    def init_prompt(self):
        if self.save_file["context"] != "" and not self.save_file["context"].startswith("This is the translation context, please refer to it: "):
            self.save_file["context"] = "This is the translation context, please refer to it: " + self.save_file["context"] + " "

        def prompt_replace(key):
            self.save_file[key] = (
                self.save_file[key]
                .replace("##src_lng##", self.save_file["src_lng"])
                .replace("##tgt_lng##", self.save_file["tgt_lng"])
                .replace("##source##", self.save_file["source"])
                .replace("##base_translation##", self.save_file["base_translation"])
                .replace("##context##", self.save_file["context"] if self.save_file["context"] != "" else "")
            )

        prompt_replace("initial_prompt")
        prompt_replace("debate_prompt")
        prompt_replace("player_meta_prompt")
        prompt_replace("moderator_meta_prompt")
        prompt_replace("base_prompt")

    def creat_agents(self):
        self.moderator = Player(model_name=self.moderator_name, name='Moderator', temperature=self.temperature, api_keys=self.api_keys, sleep_time=self.sleep_time)
        self.save_file['moderator'] = self.moderator.model_name
        self.players = [Player(model_name=self.Society[name], name=name, temperature=self.temperature, api_keys=self.api_keys, sleep_time=self.sleep_time) for name in self.Society]
        self.agents = {}
        for i, player in enumerate(self.players, 1):
            agent_name = f'agent{i}'
            self.agents[agent_name] = player
            self.save_file[agent_name] = player.model_name

    def init_agents(self):
        self.moderator.set_meta_prompt(self.save_file['moderator_meta_prompt'])
        for agent in self.agents.values():
            agent.set_meta_prompt(self.save_file['player_meta_prompt'])


        




    def round_dct(self, num: int):
        dct = {
            1: 'first', 2: 'second', 3: 'third', 4: 'fourth', 5: 'fifth', 6: 'sixth', 7: 'seventh', 8: 'eighth', 9: 'ninth', 10: 'tenth'
        }
        return dct[num]
            
    def save_file_to_json(self, id):
        now = datetime.now()
        current_time = now.strftime("%Y-%m-%d_%H:%M:%S")
        save_file_path = os.path.join(self.save_file_dir, f"{id}.json")
        
        self.save_file['end_time'] = current_time
        json_str = json.dumps(self.save_file, ensure_ascii=False, indent=4)
        with open(save_file_path, 'w') as f:
            f.write(json_str)

    def broadcast(self, msg: str):
        """Broadcast a message to all players. 
        Typical use is for the host to announce public information

        Args:
            msg (str): the message
        """
        # print(msg)
        for player in self.players:
            player.add_event(msg)

    def speak(self, speaker: str, msg: str):
        """The speaker broadcast a message to all other players. 

        Args:
            speaker (str): name of the speaker
            msg (str): the message
        """
        if not msg.startswith(f"{speaker}: "):
            msg = f"{speaker}: {msg}"
        # print(msg)
        for player in self.players:
            if player.name != speaker:
                player.add_event(msg)

    def ask_and_speak(self, player: Player):
        ans = player.ask()
        player.add_memory(ans)
        self.speak(player.name, ans)





    def run(self):
        agent_answers = {}

        for name, agent in self.agents.items():
            agent.add_event(self.save_file['initial_prompt'])
            raw_ans = agent.ask()
            agent.add_memory(raw_ans)

            json_list = extract_json_objects_resilient(raw_ans, repair_model=self.repair_model)
            if json_list:
                ans = json_list[0]
            else:
                print("[DEBUG] Could not extract valid JSON from the LLM response:")
                raise ValueError("Failed to extract valid JSON from the LLM response")

            agent_answers[name] = ans

        mod_prompt = self.save_file['moderator_prompt']

        for name in sorted(agent_answers):
            translation_raw = agent_answers[name].get("Translation")
            translation = translation_raw.strip() if isinstance(translation_raw, str) else ""

            reason_raw = agent_answers[name].get("Brief Reason")
            reason = reason_raw.strip() if isinstance(reason_raw, str) else ""

            combined = f"Translation:{translation}\nBrief Reason:{reason}\n##candidate_solutions##"
            mod_prompt = mod_prompt.replace("##candidate_solutions##", combined)

        mod_prompt = mod_prompt.replace("##candidate_solutions##", "")
        mod_prompt = mod_prompt.replace("##round##", "first")
        self.moderator.add_event(mod_prompt)
        mod_ans = self.moderator.ask()
        self.moderator.add_memory(mod_ans)

        json_list = extract_json_objects_resilient(mod_ans, repair_model=self.repair_model)
        if json_list:
            mod_ans = json_list[0]
        else:
            print("[DEBUG] Could not extract valid JSON from the LLM response:")
            print(mod_ans)
            raise ValueError("Failed to extract valid JSON from the LLM response")

        self.save_file["translations"].append(mod_ans.get("Translation", ""))

        # rounds data
        round_data = {
            "round": 0,
        }
        for name in sorted(agent_answers):
            round_data[name] = agent_answers[name]
        round_data["moderator"] = mod_ans
        self.save_file["rounds"].append(round_data)












        for round in range(self.round - 1):
            updated_answers = {}



            for name, agent in self.agents.items():
                other_translations = []
                for n in self.agents:
                    if n != name:
                        ans = agent_answers.get(n)
                        if ans is None:
                            print(f"[WARNING] agent_answers for '{n}' is None.")
                            continue
                        translation_raw = ans.get("Translation")
                        translation = translation_raw.strip() if isinstance(translation_raw, str) else ""
                        if not isinstance(translation_raw, str):
                            print(f"[WARNING] 'Translation' for agent '{n}' is not a string: {translation_raw}")

                        reason_raw = ans.get("Brief Reason")
                        reason = reason_raw.strip() if isinstance(reason_raw, str) else ""
                        if not isinstance(reason_raw, str):
                            print(f"[WARNING] 'Brief Reason' for agent '{n}' is not a string: {reason_raw}")

                        combined = f"Translation:{translation}\nBrief Reason:{reason}"
                        other_translations.append(combined)

                prompt = self.save_file['debate_prompt']
                for trans in other_translations:
                    prompt = prompt.replace("##candidate_solutions##", trans + "\n##candidate_solutions##")

                prompt = prompt.replace("##candidate_solutions##", "")
                mos_translation = mod_ans.get("Translation")
                mos_brief_reason = mod_ans.get("Brief Reason")
                if mos_translation and mos_brief_reason:
                    replacement = f"Translation: {mos_translation}, Brief Reason: {mos_brief_reason}"
                elif mos_translation:
                    replacement = f"Translation: {mos_translation}"
                elif mos_brief_reason:
                    replacement = f"Brief Reason: {mos_brief_reason}"
                else:
                    replacement = ""
                prompt = prompt.replace("##pre_mod##", replacement)



                agent.add_event(prompt)
                raw_ans = agent.ask()
                agent.add_memory(raw_ans)

                json_list = extract_json_objects_resilient(raw_ans, repair_model=self.repair_model)
                if json_list:
                    ans = json_list[0]
                else:
                    print("[DEBUG] Could not extract valid JSON from the LLM response:")
                    print(mod_ans)
                    raise ValueError("Failed to extract valid JSON from the LLM response")
                
                updated_answers[name] = ans
            agent_answers = updated_answers

            mod_prompt = self.save_file['moderator_prompt']
            for name in sorted(agent_answers):

                translation_raw = agent_answers[name].get("Translation")
                translation = translation_raw.strip() if isinstance(translation_raw, str) else ""

                reason_raw = agent_answers[name].get("Brief Reason")
                reason = reason_raw.strip() if isinstance(reason_raw, str) else ""


                combined = f"Translation:{translation}\nBrief Reason:{reason}\n##candidate_solutions##"
                mod_prompt = mod_prompt.replace("##candidate_solutions##", combined)



            mod_prompt = mod_prompt.replace("##candidate_solutions##", "")
            mod_prompt = mod_prompt.replace("##round##", self.round_dct(round + 2))
            self.moderator.add_event(mod_prompt)
            mod_ans = self.moderator.ask()
            self.moderator.add_memory(mod_ans)


            json_list = extract_json_objects_resilient(mod_ans, repair_model=self.repair_model)
            if json_list:
                mod_ans = json_list[0]
            else:
                print("[DEBUG] Could not extract valid JSON from the LLM response:")
                print(mod_ans)
                raise ValueError("Failed to extract valid JSON from the LLM response")

            self.save_file["translations"].append(mod_ans.get("Translation", ""))

            # round data
            round_data = {
                "round": round + 1,
            }
            for name in sorted(agent_answers):
                round_data[name] = agent_answers[name]
            round_data["moderator"] = mod_ans
            self.save_file["rounds"].append(round_data)


        if "Translation" in mod_ans and mod_ans["Translation"]:
            self.save_file["debate_translation"] = mod_ans["Translation"]

        for player in self.players:
            self.save_file['players'][player.name] = player.memory_lst
        self.save_file['players'][self.moderator.name] = self.moderator.memory_lst

class Zero_Shot:
    def __init__(self,
            base_name: str=None,
            temperature: float=0,
            save_file_dir: str=None,
            api_keys: dict=None,
            prompts_path: str=None,
            sleep_time: float=0,
            config_path: str=None
        ) -> None:

        if base_name is None:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            base_name = config['model_name']

        self.model_name = base_name
        self.temperature = temperature
        self.save_file_dir = save_file_dir
        self.api_keys = api_keys
        self.sleep_time = sleep_time

        # init save file
        now = datetime.now()
        current_time = now.strftime("%Y-%m-%d_%H:%M:%S")
        self.save_file = {
            'start_time': current_time,
            'end_time': '',
            'base_name': base_name,
            'temperature': temperature,
            "src_lng": "",
            "tgt_lng": "",
            'source': '',
            'reference': '',
            'base_translation': '',
            'players': {},
            "bleu_scores": [],
            "translations": [],
        }
        prompts = json.load(open(prompts_path))
        self.save_file.update(prompts)
        self.init_prompt()

        if self.save_file['base_translation'] == "":
            self.create_base()

    def init_prompt(self):
        if self.save_file["context"] != "" and not self.save_file["context"].startswith("This is a translation context, please refer to it: "):
            self.save_file["context"] = "This is a translation context, please refer to it: " + self.save_file["context"] + " "

        def prompt_replace(key):
            self.save_file[key] = (
                self.save_file[key]
                .replace("##src_lng##", self.save_file["src_lng"])
                .replace("##tgt_lng##", self.save_file["tgt_lng"])
                .replace("##source##", self.save_file["source"])
                .replace("##base_translation##", self.save_file["base_translation"])
                .replace("##context##", self.save_file["context"] if self.save_file["context"] != "" else "")
            )

        prompt_replace("base_prompt")

    def create_base(self):
        agent = Player(model_name=self.model_name, name='Baseline', temperature=self.temperature, api_keys=self.api_keys, sleep_time=self.sleep_time)
        agent.add_event(self.save_file['base_prompt'])
        base_translation = agent.ask()
        if self.save_file['tgt_lng'] == 'Japanese':
            base_translation = extract_clean_japanese_translation(base_translation)
        agent.add_memory(base_translation)

        self.save_file['base_translation'] = base_translation
        self.save_file['translations'].append(base_translation)
        self.save_file['players'][agent.name] = agent.memory_lst
          
    def save_file_to_json(self, id):
        now = datetime.now()
        current_time = now.strftime("%Y-%m-%d_%H:%M:%S")
        save_file_path = os.path.join(self.save_file_dir, f"{id}.json")
        
        self.save_file['end_time'] = current_time
        json_str = json.dumps(self.save_file, ensure_ascii=False, indent=4)
        with open(save_file_path, 'w') as f:
            f.write(json_str)

class Decouple:
    def __init__(self,
            temperature: float=0, 
            save_file_dir: str=None,
            api_keys: dict=None,
            prompts_path: str=None,
            round: int=3,
            sleep_time: float=0,
            config_path: str=None
        ) -> None:

        with open (config_path, 'r') as f:
            config = yaml.safe_load(f)
        self.repair_model = config.get('repair_model')
        self.moderator_model = config['decouple_moderator_model']
        self.base_model = config['decouple_base_model']
        self.accuracy_model = config['accuracy']
        self.fluency_model = config['fluency']
        self.term_model = config['term']
        self.style_model = config['style']

        self.temperature = temperature
        self.save_file_dir = save_file_dir
        self.api_keys = api_keys
        self.round = round
        self.sleep_time = sleep_time

        # init save file
        now = datetime.now()
        current_time = now.strftime("%Y-%m-%d_%H:%M:%S")
        self.save_file = {
            'start_time': current_time,
            'end_time': '',
            'base_name': self.base_model,
            'moderator': self.moderator_model,
            'accuracy_model': self.accuracy_model,
            'fluency_model': self.fluency_model,
            'term_model': self.term_model,
            'style_model': self.style_model,
            'temperature': temperature,
            "src_lng": "",
            "tgt_lng": "",
            'source': '',
            'reference': '',
            "bleu_scores": [],
            "translations": [],
            "rounds" : [],
            'players': {},
            "accuracy_annotations": [],
            "fluency_annotations": [],
            "term_annotations": [],
            "style_annotations": [],
        }
        prompts = json.load(open(prompts_path))
        self.save_file.update(prompts)
        self.init_prompt()

        if self.save_file['base_translation'] == "":
            print("No zero-shot base translation available; generating one.")
            self.create_base()
        self.save_file["translations"].append(self.save_file['base_translation'])
    
        # create&init agents
        self.create_agents()
        self.init_agents()

    def init_prompt(self):
        if self.save_file["context"] != "" and not self.save_file["context"].startswith("This is a translation context, please refer to it: "):
            self.save_file["context"] = "This is the translation context, please refer to it: " + self.save_file["context"] + " "

        def prompt_replace(key):
            self.save_file[key] = (
                self.save_file[key]
                .replace("##src_lng##", self.save_file["src_lng"])
                .replace("##tgt_lng##", self.save_file["tgt_lng"])
                .replace("##source##", self.save_file["source"])
                .replace("##base_translation##", self.save_file["base_translation"])
                .replace("##context##", self.save_file["context"] if self.save_file["context"] != "" else "")
            )

        prompt_replace("base_prompt")
        prompt_replace("agent_meta_prompt")
        prompt_replace("moderator_meta_prompt")
        prompt_replace("accuracy_agent") 
        prompt_replace("fluency_agent")
        prompt_replace("term_agent")
        prompt_replace("style_agent")

    def create_base(self):
        agent = Player(model_name=self.base_model, name='Baseline', temperature=self.temperature, api_keys=self.api_keys, sleep_time=self.sleep_time)
        agent.add_event(self.save_file['base_prompt'])
        base_translation = agent.ask()

        if self.save_file['tgt_lng'] == 'Japanese':
            base_translation = extract_clean_japanese_translation(base_translation)
        
        agent.add_memory(base_translation)
        self.save_file['base_translation'] = base_translation
        self.save_file['players'][agent.name] = agent.memory_lst

    def create_agents(self):
        self.moderator = Player(model_name=self.moderator_model, name='Moderator', temperature=self.temperature, api_keys=self.api_keys, sleep_time=self.sleep_time)
        self.accuracy_agent = Player(model_name=self.accuracy_model, name='Accuracy', temperature=self.temperature, api_keys=self.api_keys, sleep_time=self.sleep_time)
        self.fluency_agent = Player(model_name=self.fluency_model, name='Fluency', temperature=self.temperature, api_keys=self.api_keys, sleep_time=self.sleep_time)
        self.term_agent = Player(model_name=self.term_model, name='Term', temperature=self.temperature, api_keys=self.api_keys, sleep_time=self.sleep_time)
        self.style_agent = Player(model_name=self.style_model, name='Style', temperature=self.temperature, api_keys=self.api_keys, sleep_time=self.sleep_time)
        self.players = [self.accuracy_agent, self.fluency_agent, self.term_agent, self.style_agent]

        self.save_file['players']['Moderator'] = self.moderator.model_name
        self.save_file['players']['Accuracy'] = self.accuracy_agent.model_name
        self.save_file['players']['Fluency'] = self.fluency_agent.model_name
        self.save_file['players']['Term'] = self.term_agent.model_name
        self.save_file['players']['Style'] = self.style_agent.model_name

    def init_agents(self):
        # start: set meta prompt
        self.moderator.set_meta_prompt(self.save_file['moderator_meta_prompt'])
        self.accuracy_agent.set_meta_prompt(self.save_file['agent_meta_prompt'])
        self.fluency_agent.set_meta_prompt(self.save_file['agent_meta_prompt'])
        self.term_agent.set_meta_prompt(self.save_file['agent_meta_prompt'])
        self.style_agent.set_meta_prompt(self.save_file['agent_meta_prompt'])

    def round_dct(self, num: int):
        dct = {
            1: 'first', 2: 'second', 3: 'third', 4: 'fourth', 5: 'fifth', 6: 'sixth', 7: 'seventh', 8: 'eighth', 9: 'ninth', 10: 'tenth'
        }
        return dct[num]
            
    def save_file_to_json(self, id):
        now = datetime.now()
        current_time = now.strftime("%Y-%m-%d_%H:%M:%S")
        save_file_path = os.path.join(self.save_file_dir, f"{id}.json")
        
        self.save_file['end_time'] = current_time
        json_str = json.dumps(self.save_file, ensure_ascii=False, indent=4)
        with open(save_file_path, 'w') as f:
            f.write(json_str)

    def broadcast(self, msg: str):
        """Broadcast a message to all players. 
        Typical use is for the host to announce public information

        Args:
            msg (str): the message
        """
        # print(msg)
        for player in self.players:
            player.add_event(msg)

    def speak(self, speaker: str, msg: str):
        """The speaker broadcast a message to all other players. 

        Args:
            speaker (str): name of the speaker
            msg (str): the message
        """
        if not msg.startswith(f"{speaker}: "):
            msg = f"{speaker}: {msg}"
        # print(msg)
        for player in self.players:
            if player.name != speaker:
                player.add_event(msg)

    def ask_and_speak(self, player: Player):
        ans = player.ask()
        player.add_memory(ans)
        self.speak(player.name, ans)

    def run(self):
        def fill_template(template, warn_missing=False, **kwargs):
            for k, v in kwargs.items():
                if v is not None:
                    template = template.replace(f"##{k}##", str(v))

            if warn_missing:
                import re
                missing = re.findall(r"##(\w+)##", template)
                if missing:
                    print(f"[WARN] These fields are still missing in template: {missing}")

            return template


        def run_agent(agent, prompt_template_key, target_segment, save_key):
            prompt = fill_template(self.save_file[prompt_template_key], target_segment=target_segment)
            agent.add_event(prompt)
            ans = agent.ask()
            agent.add_memory(ans)
            self.save_file[save_key].append(ans)
            return ans

        accuracy_ans = run_agent(self.accuracy_agent, 'accuracy_agent', self.save_file['base_translation'], 'accuracy_annotations')
        fluency_ans  = run_agent(self.fluency_agent, 'fluency_agent', self.save_file['base_translation'], 'fluency_annotations')
        term_ans     = run_agent(self.term_agent, 'term_agent', self.save_file['base_translation'], 'term_annotations')
        style_ans    = run_agent(self.style_agent, 'style_agent', self.save_file['base_translation'], 'style_annotations')




        moderator_prompt = fill_template(
            self.save_file["moderator_prompt"],
            round="first",
            accuracy_annotation=accuracy_ans,
            fluency_annotation=fluency_ans,
            term_annotation=term_ans,
            style_annotation=style_ans,
        )

        self.moderator.add_event(moderator_prompt)
        mod_ans = self.moderator.ask()
        self.moderator.add_memory(mod_ans)

        json_list = extract_json_objects_resilient(mod_ans, repair_model=self.repair_model)
        if json_list:
            mod_ans = json_list[0]
        else:
            print("[DEBUG] Could not extract valid JSON from the LLM response:")
            print(mod_ans)
            raise ValueError("Failed to extract valid JSON from the LLM response")


        if "Translation" in mod_ans and mod_ans["Translation"]:
            mod_ans["Translation"] = mod_ans["Translation"].strip("「」")
            if self.save_file["tgt_lng"] == 'Japanese':
                mod_ans["Translation"] = extract_clean_japanese_translation(mod_ans["Translation"])

            self.save_file["translations"].append(mod_ans["Translation"])

        round_data = {
            "round": 0,
            "accuracy": accuracy_ans,
            "fluency": fluency_ans,
            "term": term_ans,
            "style": style_ans,
            "moderator": mod_ans,
        }
        self.save_file["rounds"].append(round_data)

        for round in range(self.round - 1):
            accuracy_ans = run_agent(self.accuracy_agent, 'accuracy_agent', self.save_file['translations'][round+1], 'accuracy_annotations')
            fluency_ans  = run_agent(self.fluency_agent, 'fluency_agent', self.save_file['translations'][round+1], 'fluency_annotations')
            term_ans     = run_agent(self.term_agent, 'term_agent', self.save_file['translations'][round+1], 'term_annotations')
            style_ans    = run_agent(self.style_agent, 'style_agent', self.save_file['translations'][round+1], 'style_annotations')

            moderator_prompt = fill_template(
                self.save_file["moderator_prompt"],
                round=self.round_dct(round + 2),
                accuracy_annotation=accuracy_ans,
                fluency_annotation=fluency_ans,
                term_annotation=term_ans,
                style_annotation=style_ans,
            )


            self.moderator.add_event(moderator_prompt)
            mod_ans = self.moderator.ask()
            self.moderator.add_memory(mod_ans)
            json_list = extract_json_objects_resilient(mod_ans, repair_model=self.repair_model)
            if json_list:
                mod_ans = json_list[0]
            else:   
                print("[DEBUG] Could not extract valid JSON from the LLM response:")
                print(mod_ans)
                raise ValueError("Failed to extract valid JSON from the LLM response")
            if "Translation" in mod_ans and mod_ans["Translation"]:
                mod_ans["Translation"] = mod_ans["Translation"].strip("「」")
                if self.save_file["tgt_lng"] == 'Japanese':
                    mod_ans["Translation"] = extract_clean_japanese_translation(mod_ans["Translation"])
                self.save_file["translations"].append(mod_ans["Translation"])

            # round data
            round_data = {
                "round": round + 1,
                "accuracy": accuracy_ans,
                "fluency": fluency_ans,   
                "term": term_ans,
                "style": style_ans,
                "moderator": mod_ans,
            }
            self.save_file["rounds"].append(round_data)       
        self.save_file['players'] = {
            'accuracy_agent': self.accuracy_agent.memory_lst,
            'fluency_agent': self.fluency_agent.memory_lst,
            'term_agent': self.term_agent.memory_lst,
            'style_agent': self.style_agent.memory_lst,
            'moderator': self.moderator.memory_lst
        }
