import openai
from pynput import keyboard

# Replace 'YOUR_API_KEY' with your actual API key from OpenAI
api_key = 'YOUR_API_KEY'
openai.api_key = api_key

def get_completion(prompt, model="gpt-3.5-turbo"):

    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(model=model, messages=messages, temperature=0)

    return response.choices[0].message["content"]

def get_input(user_size=0):
    last_state = user_size
    user = input()
    user_size = len(user)
    if user_size > last_state:
        q1 = "<What are the five most common words that start with " + user + ">"
        response = get_completion(q1)
        print(response)
    return user_size

user_size = 0
while user_size < 3:
    user_size = get_input()
    
#%% V2

import openai
from pynput import keyboard

# Replace 'YOUR_API_KEY' with your actual API key from OpenAI
api_key = 'YOUR_API_KEY'
openai.api_key = api_key

def get_completion(prompt, model="gpt-3.5-turbo"):

    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(model=model, messages=messages, temperature=0)

    return response.choices[0].message["content"]

input_ = ''

while len(input_) < 14:
    print(len(input_))
    with keyboard.Events() as events:
        event = events.get(1e6)
        input_ = input_ + str(event.key)
    print(input_)
        #q1 = "<What are the five most common words that start with " + input_ + ">"
        #print(get_completion(q1))
        
#%% V3 Implements threading

import threading
import time
import openai

openai.api_key = 'YOUR_API_KEY'

def get_completion(prompt, model="gpt-3.5-turbo"):
    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(model=model, messages=messages, temperature=0)
    return response.choices[0].message["content"]

def keyboard_input_thread():
    while True:
        user_input = input("Enter input: ")
        update_prompt(user_input)

def update_prompt(user_input):
    global prompt
    prompt += user_input
    print(get_completion("<What are the five most common words that start with " + prompt + ">"))

prompt = ""

keyboard_thread = threading.Thread(target=keyboard_input_thread)
keyboard_thread.daemon = True
keyboard_thread.start()

try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    pass

Add an if statement for when chatGPT returns " There are no commonly used words in the English language that start with "wj." "
 
