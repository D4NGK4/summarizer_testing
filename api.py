from typing import List, Tuple, Optional
import tiktoken
from tqdm import tqdm
from openai import OpenAI
import os
import json
from dotenv import load_dotenv
import random

load_dotenv()


keyA = (os.environ.get('API_KEY'))
keyB = (os.environ.get('CLYDE_KEY'))

counter = 0

bol = False


def my_function(bol):
    if bol == False:
        client = os.environ.get('API_KEY')
        print(client)
        return not bol
    
    if bol == True:
        client = os.environ.get('CLYDE_KEY')
        print(client)
        return not bol
    
while counter < 6:
    counter += 1
    my_function(bol)


