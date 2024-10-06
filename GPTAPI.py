# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# """
# Created on Tue Sep 19 10:59:31 2023

# @author: alossius
# """

import threading
from pynput import keyboard
import openai
import time

openai.api_key = 'UNIQUE API KEY' # message me for key

def get_completion(prompt, model="gpt-3.5-turbo"):
    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(model=model, messages=messages, temperature=0)
    return response.choices[0].message["content"] + "\n"

def on_press(key):
    try:
        global prompt
        if hasattr(key, 'char'):
            # Append the pressed key to the prompt
            prompt += key.char
            print('\n' + get_completion("<What are the five most common words that start with " + prompt + ">"))
            print('Enter next character: ')
    except AttributeError:
        pass

def keyboard_input_thread():
    with keyboard.Listener(on_press=on_press) as listener:
        print('Enter first character: ')
        listener.join()

prompt = ""

keyboard_thread = threading.Thread(target=keyboard_input_thread)
keyboard_thread.daemon = True
keyboard_thread.start()

try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    pass

