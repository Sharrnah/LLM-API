import os
import glob
import re
from datetime import datetime
import threading

from llama_cpp import Llama

import chat_history

# Try to get the model file name from the environment variable
env_model_file = os.environ.get('MODEL_FILE')

# Directories to search for the .gguf file
directories = ["/root/.cache/llama2/", "cache/llama2/"]

# Initialize transformer_model_path to None
transformer_model_path = None

# If the environment variable is set, use it to construct the model path
if env_model_file:
    for directory in directories:
        potential_path = os.path.join(directory, env_model_file)
        if os.path.exists(potential_path):
            transformer_model_path = potential_path
            break  # Exit the loop if the model file is found

# If transformer_model_path is still None, search for any .gguf file
if transformer_model_path is None:
    for directory in directories:
        # Use glob to find all .gguf files in the directory
        gguf_files = glob.glob(os.path.join(directory, "*.gguf"))

        # If any .gguf files are found, take the first one
        if gguf_files:
            transformer_model_path = gguf_files[0]
            break  # Exit the loop if a .gguf file is found

# Check if a .gguf file was found
if transformer_model_path is None:
    print(".gguf file not found in any of the specified directories.")
else:
    print(f".gguf file found: {transformer_model_path}")

############################
# force stop generating string
STOP_GENERATING_STRING = "[end of text]"

# 'max_new_tokens': 512
# Llama2-Chat config
config = {'max_new_tokens': 384, 'repetition_penalty': 1.1, 'temperature': 0.7, 'context_length': 3072,
          'stop': [STOP_GENERATING_STRING], 'threads': int(os.cpu_count())}

# CodeLlama config
#config = {'max_new_tokens': 1024, 'repetition_penalty': 1.1, 'temperature': 0.1, 'context_length': 3072, 'threads': int(os.cpu_count())}

# load model
print("loading " + transformer_model_path + " ... using " + str(os.cpu_count()) + " threads.")
llm = Llama(model_path=transformer_model_path, n_gpu_layers=30, n_ctx=config['context_length'])


instructions = {
    "_": {
        'init_prompt': "[INST] <<SYS>>\nYou are a helpful, respectful, honest Assistant. your name is {ai_name}. current time is {current_day}, {current_datetime}. Only tell the date and time if asked. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Ensure that your responses are socially unbiased and positive in nature. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, Don't share false information. Keep the answers short.{summary}\n<</SYS>>\n{history}{name}{prompt}" + STOP_GENERATING_STRING + "[/INST]\n",
        'result_replacements': {
            STOP_GENERATING_STRING: "",
        },
        'ai_name': "Assistant",
        'save_history': False,
        'generate_summary_on_full_history': False,
        'remove_emotions_from_history': False,
        'instruct_tags_type': "llama2",
        'communication_type': "multi_user_chat",
    },
    # CodeLlama (Alpaca/Vicuna instruction format) model template (https://huggingface.co/Phind/Phind-CodeLlama-34B-v2 , https://huggingface.co/TheBloke/Phind-CodeLlama-34B-v2-GGUF)
    "coding_llm": {
        'init_prompt': "### System Prompt\nYou are an intelligent programming assistant.\n\n### User Message\n{prompt}\n\n### Assistant\n",
        'result_replacements': {},
        'ai_name': "Assistant",
        'save_history': False,
        'generate_summary_on_full_history': False,
        'remove_emotions_from_history': False,
        'instruct_tags_type': "",
        'communication_type': "",
    }
}

# CHAT_MAX_HISTORY_ENTRIES should at least have 2 more than CHAT_MAX_DETAILED_HISTORY
# (because chat and answer are added before summarization)
CHAT_MAX_HISTORY_ENTRIES = 7
CHAT_MAX_DETAILED_HISTORY = 4
chat_manager = chat_history.ChatManager()
for chat_manager_instruction in instructions.keys():
    chat_manager.initialize_chat(chat_manager_instruction, max_entries=CHAT_MAX_HISTORY_ENTRIES)
    chat_manager.load_history_from_file(chat_manager_instruction)


def remove_llama2_instruct_tags(text):
    # Use regular expression to remove everything between [INST] and [/INST], including the tags themselves
    text = re.sub(r'\[INST\].*?\[/INST\]\n?', '', text, flags=re.DOTALL)

    text = text.replace(STOP_GENERATING_STRING, "")
    text = text.replace("[INST]", "").replace("[/INST]", "")
    text = text.replace("<<SYS>>", "").replace("<</SYS>>", "")

    return text


def remove_emotions(text):
    # This regular expression matches any string that starts with an asterisk (*) and ends with an asterisk (*)
    # and removes it from the text. (like *ears perk up* and similar)
    return re.sub(r'(?<!\*)\*.*?\*(?!\*)', '', text).strip()


def replace_strings_in_result(text, instruction='tiger'):
    result_replacements = instructions[instruction]['result_replacements']

    for original, new in result_replacements.items():
        text = text.replace(original, new)
    return text


def add_summary_to_chat_history(instruction_name='_'):
    chat_manager.generate_summary(instruction_name)
    chat_manager.clear_chat(instruction_name, retain_count=CHAT_MAX_DETAILED_HISTORY)

    chat_manager.save_history_to_file(instruction_name)


def message(text, name='User', instruction='_', disable_history=False):
    global llm

    name_str = ''
    if name != '':
        name_str = name + ": "

    # Get the current date, time, and day of the week
    current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    current_day = datetime.now().strftime("%A")

    # Get AI name of instructions set
    ai_name = instructions[instruction]['ai_name']

    # remove instruct tags from user input to prevent issues.
    if instructions[instruction]['instruct_tags_type'] == 'llama2':
        text = remove_llama2_instruct_tags(text)

    # Fetch chat history and summary for AI memory
    summary = ""
    history = ""
    if not disable_history and instructions[instruction]['save_history']:
        summary = chat_manager.get_summary(instruction)
        if summary != "":
            summary = " Summary of previous messages: " + summary
        history = chat_manager.get_all_messages_string(instruction, ai_name=ai_name, stop=STOP_GENERATING_STRING)
        if history != "":
            history = history + "\n"

    # format prompt to fit instruction prompt template
    prompt = instructions[instruction]['init_prompt'].format(
        ai_name=ai_name,
        summary=summary,
        history=history,
        prompt=text, name=name_str, current_day=current_day, current_datetime=current_datetime
    )

    # Call the AI model
    try:
        answer_dict = llm(prompt,
                          max_tokens=config['max_new_tokens'],
                          stop=[STOP_GENERATING_STRING],
                          echo=False,
                          temperature=config['temperature'],
                          repeat_penalty=config['repetition_penalty'],
                          )
        print(answer_dict)
        answer = answer_dict['choices'][0]['text']
    except Exception as e:
        print(e)
        return ""

    # cleanup AI answer according to instruction config
    if instructions[instruction]['instruct_tags_type'] == 'llama2':
        answer = remove_llama2_instruct_tags(answer)  # these should never be in the answer text
    answer = replace_strings_in_result(answer)

    # remove chat name from the beginning of the answer (and possible from empty name remaining ':')
    if instructions[instruction]['communication_type'] == 'multi_user_chat':
        while answer.startswith(ai_name):
            answer = answer[len(ai_name):]
        answer = answer.strip()
        while answer.startswith(":"):
            answer = answer[len(":"):]

    # trim spaces from answer
    answer = answer.strip()

    # add new entries to chat history and generate new summary if needed
    if not disable_history and instructions[instruction]['save_history']:
        history_answer = answer.replace("\n", " ")
        if instructions[instruction]['remove_emotions_from_history']:
            history_answer = remove_emotions(answer.replace("\n", " "))
        chat_manager.add_message(instruction, name, text.replace("\n", " "))
        chat_manager.add_message(instruction,
                                 instructions[instruction]['ai_name'],
                                 history_answer
                                 )
        # generate new summary if chat history is full (in a background thread)
        if instructions[instruction]['generate_summary_on_full_history'] and chat_manager.is_full(instruction):
            threading.Thread(target=add_summary_to_chat_history, args=(instruction,)).start()

        chat_manager.save_history_to_file(instruction)

    print("Chat answer:" + answer)

    return answer


async def message_stream(text, name='User', instruction='_', disable_history=False):
    global llm

    name_str = ''
    if name != '':
        name_str = name + ": "

    # Get the current date, time, and day of the week
    current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    current_day = datetime.now().strftime("%A")

    # Get AI name of instructions set
    ai_name = instructions[instruction]['ai_name']

    # remove instruct tags from user input to prevent issues.
    if instructions[instruction]['instruct_tags_type'] == 'llama2':
        text = remove_llama2_instruct_tags(text)

    # Fetch chat history and summary for AI memory
    summary = ""
    history = ""
    if not disable_history and instructions[instruction]['save_history']:
        summary = chat_manager.get_summary(instruction)
        if summary != "":
            summary = " Summary of previous messages: " + summary
        history = chat_manager.get_all_messages_string(instruction, ai_name=ai_name, stop=STOP_GENERATING_STRING)
        if history != "":
            history = history + "\n"

    # format prompt to fit instruction prompt template
    prompt = instructions[instruction]['init_prompt'].format(
        ai_name=ai_name,
        summary=summary,
        history=history,
        prompt=text, name=name_str, current_day=current_day, current_datetime=current_datetime
    )

    # Call the AI model
    answer_text = ""
    try:
        text_stream = llm(prompt, stream=True,
                          max_tokens=config['max_new_tokens'],
                          stop=[STOP_GENERATING_STRING],
                          echo=False,
                          temperature=config['temperature'],
                          repeat_penalty=config['repetition_penalty'],
                          )
        for answer in text_stream:
            answer_text += answer["choices"][0]["text"]
            yield answer["choices"][0]["text"]

        # add new entries to chat history and generate new summary if needed
        if not disable_history and instructions[instruction]['save_history']:
            history_answer = answer_text.replace("\n", " ")
            if instructions[instruction]['remove_emotions_from_history']:
                history_answer = remove_emotions(answer_text)
            chat_manager.add_message(instruction, name, text)
            chat_manager.add_message(instruction,
                                     instructions[instruction]['ai_name'],
                                     history_answer
                                     )
            # generate new summary if chat history is full (in a background thread)
            if instructions[instruction]['generate_summary_on_full_history'] and chat_manager.is_full(instruction):
                threading.Thread(target=add_summary_to_chat_history, args=(instruction,)).start()

            chat_manager.save_history_to_file(instruction)

    except Exception as e:
        print(e)


def inject_memory(text, name='AI', instruction='_'):
    if not instructions[instruction]['save_history']:
        return None

    usr_name = name
    ai_name = instructions[instruction]['ai_name']
    if name.lower() == 'AI'.lower():
        usr_name = ai_name

    text = text.format(AI=ai_name)

    history_answer = text.replace("\n", " ")
    if instructions[instruction]['remove_emotions_from_history']:
        history_answer = remove_emotions(text.replace("\n", " "))

    chat_manager.add_message(instruction,
                             usr_name,
                             history_answer
                             )
    if chat_manager.is_full(instruction):
        threading.Thread(target=add_summary_to_chat_history, args=(instruction,)).start()
    chat_manager.save_history_to_file(instruction)

    return "SUCCESS"
