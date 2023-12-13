# ============================================================
# Adds Large Language Model support over API to Whispering Tiger
# answers to questions using speech to text
# V1.0.1
#
# See https://github.com/Sharrnah/whispering
# ============================================================
#
import json
from urllib.parse import urlencode

import requests

import Plugins
import VRC_OSCLib
import settings
import websocket
from Models.TextTranslation import texttranslate

PROMPT_FORMATTING = {
    "question": ["about ", "across ", "after ", "against ", "along ", "am ", "amn't ", "among ", "are ", "aren't ", "around ", "at ", "before ", "behind ", "between ",
                 "beyond ", "but ", "by ", "can ", "can't ", "concerning ", "could ", "couldn't ", "despite ", "did ", "didn't ", "do ", "does ", "doesn't ", "don't ",
                 "down ", "during ", "except ", "following ", "for ", "from ", "had ", "hadn't ", "has ", "hasn't ", "have ", "haven't ", "how ", "how's ", "in ",
                 "including ", "into ", "is ", "isn't ", "like ", "may ", "mayn't ", "might ", "mightn't ", "must ", "mustn't ", "near ", "of ", "off ", "on ", "out ",
                 "over ", "plus ", "shall ", "shan't ", "should ", "shouldn't ", "since ", "through ", "throughout ", "to ", "towards ", "under ", "until ", "up ", "upon ",
                 "was ", "wasn't ", "were ", "weren't ", "what ", "what's ", "when ", "when's ", "where ", "where's ", "which ", "which's ", "who ", "who's ", "why ",
                 "why's ", "will ", "with ", "within ", "without ", "won't ", "would ", "wouldn't "],
    "command": ["ai? ", "ai. ", "ai ", "a.i. ", "ai, ", "ai! ", "artificial intelligence"],
}


class LlmApiPlugin(Plugins.Base):
    def __init__(self):
        self.orig_osc_auto_processing_enabled = settings.GetOption("osc_auto_processing_enabled")
        self.orig_tts_answer = settings.GetOption("tts_answer")
        super().__init__()
    
    def init(self):
        # prepare all possible settings
        self.init_plugin_settings(
            {
                # General
                "osc_prefix": "AI: ",
                "osc_enabled": True,
                "osc_notify": True,
                "osc_delay": {"type": "slider", "min": 1, "max": 60, "step": 1, "value": 10},
                "tts_answer": False,
                "auth_token": "",
                "instruction_name": "_",
                "api_url": "",
                "only_respond_question_commands": False,
                "translate_to_speaker_language": False,
            },
            settings_groups={
                "General": ["osc_prefix", "osc_enabled", "osc_notify", "tts_answer", "api_url", "auth_token", "only_respond_question_commands", "translate_to_speaker_language", "osc_delay", "instruction_name"],
            }
        )

        if self.is_enabled(False):
            # disable OSC processing so the LLM can take it over:
            settings.SetOption("osc_auto_processing_enabled", False)
            # disable TTS so the LLM can take it over:
            settings.SetOption("tts_answer", False)
            # disable websocket final messages processing so the LLM can take it over:
            settings.SetOption("websocket_final_messages", False)
        else:
            settings.SetOption("osc_auto_processing_enabled", self.orig_osc_auto_processing_enabled)
            settings.SetOption("tts_answer", self.orig_tts_answer)
            settings.SetOption("websocket_final_messages", True)

    def _generate_chat_response(self, text_prompt, name, instruction_name, api_url, auth_token=""):
        # Base URL - ensure this URL is correct
        base_url = api_url

        # Add query parameters
        params = {
            'text_prompt': text_prompt,
            'name': name,
            'instruction_name': instruction_name,
            'disable_history': True,
        }
        url_with_params = f"{base_url}?{urlencode(params)}"

        # Create headers with the authentication token
        headers = {
            'X-Auth-Token': auth_token
        }

        # Make the HTTP POST request
        try:
            response = requests.post(url_with_params, headers=headers)
            response.raise_for_status()  # This will raise an HTTPError if the HTTP request returned an unsuccessful status code
            return response.text, None
        except requests.HTTPError as http_err:
            print(f"HTTP error occurred: {http_err}")  # Python 3.6+
            return None, http_err
        except Exception as err:
            print(f"Other error occurred: {err}")  # Python 3.6+
            return None, err

    def send_message(self, text, answer, result_obj):
        osc_ip = settings.GetOption("osc_ip")
        osc_address = settings.GetOption("osc_address")
        osc_port = settings.GetOption("osc_port")
        llm_osc_prefix = self.get_plugin_setting("osc_prefix", "AI: ")
        osc_notify = self.get_plugin_setting("osc_notify")
        osc_delay = self.get_plugin_setting("osc_delay")

        result_obj["type"] = "llm_answer"
        try:
            print("LLM Answer: " + answer)
        except:
            print("LLM Answer: ???")

        if self.get_plugin_setting("osc_enabled", True) and answer != text and osc_ip != "0":
            VRC_OSCLib.Chat_chunks(llm_osc_prefix + answer,
                                   nofify=osc_notify, address=osc_address, ip=osc_ip, port=osc_port,
                                   chunk_size=144, delay=osc_delay,
                                   initial_delay=osc_delay,
                                   convert_ascii=settings.GetOption("osc_convert_ascii"))

        websocket.BroadcastMessage(json.dumps(result_obj))

    def stt(self, text, result_obj):
        if self.is_enabled(False):
            if not self.get_plugin_setting("only_respond_question_commands") or (self.get_plugin_setting("only_respond_question_commands") and (("?" in text.strip().lower() and any(ele in text.strip().lower() for ele in PROMPT_FORMATTING['question'])) or
                                                                               any(ele in text.strip().lower() for ele in PROMPT_FORMATTING['command']))):
                instruction_name = self.get_plugin_setting("instruction_name")
                predicted_text, _ = self._generate_chat_response(
                    text_prompt=text,
                    name="User",
                    instruction_name=instruction_name,
                    api_url=self.get_plugin_setting("api_url"),
                    auth_token=self.get_plugin_setting("auth_token")
                )

                if self.get_plugin_setting("translate_to_speaker_language", False):
                    target_lang = result_obj['language']
                    print("Translating to " + target_lang)
                    predicted_text, txt_from_lang, txt_to_lang = texttranslate.TranslateLanguage(predicted_text, "auto",
                                                                                                 target_lang,
                                                                                                 False, True)
                result_obj['llm_answer'] = predicted_text

                print("llm_answer: ", predicted_text)

                self.send_message(text, predicted_text, result_obj)

                if self.get_plugin_setting("tts_answer"):
                    self.send_tts(predicted_text)
        return

    def send_tts(self, text):
        device_index = settings.GetOption("device_out_index")
        if device_index is None or device_index == -1:
            device_index = settings.GetOption("device_default_out_index")

        for plugin_inst in Plugins.plugins:
            if hasattr(plugin_inst, 'tts'):
                plugin_inst.tts(text, device_index, None, False)

    def on_enable(self):
        self.init()
        pass

    def on_disable(self):
        self.init()
        pass
