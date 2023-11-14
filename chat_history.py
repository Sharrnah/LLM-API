import threading
import json
import os

import summary_generator


class ChatManager:
    def __init__(self):
        """Initialize a new ChatManager to manage multiple chat histories."""
        self.chats = {}
        self.locks = {}  # Separate lock for each chat_key

    def _get_lock(self, chat_key):
        """Get the lock for a given chat_key, create one if it doesn't exist."""
        if chat_key not in self.locks:
            self.locks[chat_key] = threading.Lock()
        return self.locks[chat_key]
    
    def initialize_chat(self, chat_key, max_entries=None):
        """Initialize a new chat history with an optional max_entries."""
        if chat_key not in self.chats:
            self.chats[chat_key] = {"messages": [], "summary": "", "max_entries": max_entries}

    def add_message(self, chat_key, name, text):
        """Add a new message to the specified chat history."""
        # If chat_key does not exist, initialize it with no max_entries by default
        if chat_key not in self.chats:
            self.initialize_chat(chat_key)

        message = {
            "name": name,
            "text": text
        }
        self.chats[chat_key]['messages'].append(message)

        # If we exceed the maximum number of entries, remove the oldest message
        max_entries = self.chats[chat_key]['max_entries']
        current_length = len(self.chats[chat_key]['messages'])
        if max_entries is not None and current_length > max_entries:
            excess = current_length - max_entries
            self.chats[chat_key]['messages'] = self.chats[chat_key]['messages'][excess:]

    def get_messages(self, chat_key):
        """Return the list of messages for the specified chat_key."""
        return self.chats.get(chat_key, {}).get('messages', [])

    def get_all_messages_string(self, chat_key, ai_name='Assistant', stop='[end of text]'):
        """ Concatenate all messages in the format "Name: text" """
        if chat_key not in self.chats:
            return ""
        # join all messages in the format "Name: Text" but when user is ai_name, add [end of text] to the end.
        return "\n".join([
            f"{message['name']}: {message['text']}{' '+stop if message['name'] == ai_name else ''}"
            for message in self.chats[chat_key]['messages']
        ])
        #return "\n".join([f"{message['name']}: {message['text']}" for message in self.chats[chat_key]['messages']])

    def get_summary(self, chat_key):
        """Return the summary for the specified chat_key."""
        return self.chats.get(chat_key, {}).get('summary', "")

    def clear_chat(self, chat_key, retain_count=0):
        """Clear all messages for the specified chat_key."""
        if chat_key in self.chats:
            if retain_count > 0:
                self.chats[chat_key]['messages'] = self.chats[chat_key]['messages'][-retain_count:]
            else:
                self.chats[chat_key]['messages'] = []

    def clear_summary(self, chat_key):
        """Clear the summary for the specified chat_key."""
        if chat_key in self.chats:
            self.chats[chat_key]['summary'] = ""

    def generate_summary(self, chat_key):
        if chat_key not in self.chats:
            return

        # Concatenate all messages in the format "Name: text"
        text_to_summarize = self.get_summary(chat_key)
        text_to_summarize += "\n\n"+self.get_all_messages_string(chat_key, stop='')

        print("generating summary for " + chat_key)
        summary = summary_generator.summarize(text_to_summarize)
        # Set the summary for the specified chat_key.
        self.chats[chat_key]['summary'] = summary
        return summary

    def get_current_size(self, chat_key):
        """Return the current number of messages for a given chat_key."""
        if chat_key not in self.chats:
            return 0
        return len(self.chats[chat_key]['messages'])

    def get_max_size(self, chat_key):
        """Return the max_entries value for a given chat_key."""
        if chat_key not in self.chats:
            return 0
        return self.chats[chat_key]['max_entries']

    def is_full(self, chat_key):
        """Check if the list of messages for a given chat_key has reached its max_entries."""
        if chat_key not in self.chats:
            return False
        max_entries = self.get_max_size(chat_key)
        current_entries = self.get_current_size(chat_key)
        return current_entries >= max_entries

    def save_history_to_file(self, chat_key, directory='chat_histories'):
        """Save individual chat history to a JSON file based on the chat_key."""
        with self._get_lock(chat_key):
            # Ensure the directory exists
            if not os.path.exists(directory):
                os.makedirs(directory)

            filename = os.path.join(directory, f"{chat_key}.json")
            try:
                with open(filename, 'w') as file:
                    json.dump(self.chats.get(chat_key, {}), file)
            except (IOError, PermissionError) as e:
                print(f"Error saving chat '{chat_key}' to file '{filename}': {e}")

    def load_history_from_file(self, chat_key, directory='chat_histories'):
        """Load individual chat history from a JSON file based on the chat_key."""
        filename = os.path.join(directory, f"{chat_key}.json")
        try:
            with open(filename, 'r') as file:
                loaded_chat = json.load(file)
                # Preserve the current max_entries if it exists
                if chat_key in self.chats:
                    loaded_chat['max_entries'] = self.chats[chat_key]['max_entries']
                self.chats[chat_key] = loaded_chat
        except FileNotFoundError:
            print(f"File for chat '{chat_key}' not found in '{directory}'. Starting with an empty chat history.")
        except (IOError, PermissionError) as e:
            print(f"Error loading chat '{chat_key}' from file '{filename}': {e}")
