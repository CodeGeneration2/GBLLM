# -*- coding: utf-8 -*-
# print(f"### :\n{}")
# #####################################################################################################################🔖💡✅🟨

# import os
# import re
# import sys
import time
# import random
# import logging
# import argparse
# import jsonlines
# import numpy as np
# import multiprocessing
from tqdm import tqdm
from time import sleep
from copy import deepcopy
import statistics
from transformers import pipeline, BitsAndBytesConfig
import torch
from torch.amp import autocast

# import google.generativeai as genai
from openai import OpenAI

# Install google-genai
from google import genai
from google.genai import types

# import json
from pprint import pprint


# #####################################################################################################################🔖💡✅🟨❌
GPU_ID = 0
CODELLAMA_API_KEYS = [
]

GEMINI_API_KEYS = [
]

CHATGPT_API_KEYS = [
]

DEEPSEEK_API_KEYS = [
]


# #####################################################################################################################🔖💡🟨✅❌
INPUT_ROLE_PLAY = "You are an animal joke master."
INPUT_PROBLEM_TEXT = "Tell a joke"


# #####################################################################################################################🔖💡🟨✅❌
def main(model_id=1):
    if model_id == 1:
        model_response_text_list = codellama_deepinfra_func(
            key_index=0, 
            model_name="CodeLlama-34b-Instruct-hf",   # CodeGeneration2/CodeLlama-34b-Instruct-hf
            input_role_play=INPUT_ROLE_PLAY, 
            input_problem_text=INPUT_PROBLEM_TEXT, 
            num_generations=3, 
            temperature=0.7, 
            max_length=1024, 
        )
        print(f"\n\n###================== Model Response Text List: \n{model_response_text_list}")
        print(f"\n\n###================== Model Response Text List[0]: \n{model_response_text_list[0]}")
        print(f"\n\n###================== Model Response Text List[1]: \n{model_response_text_list[1]}")
        print(f"\n\n###================== Model Response Text List[2]: \n{model_response_text_list[2]}")


    elif model_id == 2:
        model_response_text_list, avg_log_prob_list = gemini_official_func(
            key_index=0, 
            model_name="gemini-2.5-flash", 
            input_role_play=INPUT_ROLE_PLAY, 
            input_problem_text=INPUT_PROBLEM_TEXT, 
            total_messages=[],
            num_generations=1, 
            temperature=0, 
            max_length="No Limit", 
            thinking_budget=-404,      # -1: Enable dynamic thinking  0: Disable thinking  >0: Fixed thinking budget
            return_log_probs=False,
            is_output_prompt=True,
        )
        print(f"\n\n###================== Model Response Text List: \n{model_response_text_list}")
        print(f"\n\n###================== Model Response Text List[0]: \n{model_response_text_list[0]}")
        # print(f"\n\n###================== Model Response Text List[1]: \n{model_response_text_list[1]}")
        print(f"\n\n###================== Average Log Probability List: \n{avg_log_prob_list}")


    elif model_id == 5:
        model_response_text_list, avg_log_prob_list = chatgpt_func(
            platform="OpenAI_Official",                 
            key_index=0, 
            model_name="gpt-3.5-turbo-0125",    
            input_role_play=INPUT_ROLE_PLAY, 
            input_problem_text=INPUT_PROBLEM_TEXT, 
            total_messages=[],
            num_generations=2, 
            temperature=0.7, 
            max_length=1024, 
            return_log_probs=True,
            is_output_prompt=True,
        )
        print(f"\n\n###================== Model Response Text List: \n{model_response_text_list}")
        print(f"\n\n###================== Model Response Text List[0]: \n{model_response_text_list[0]}")
        print(f"\n\n###================== Model Response Text List[1]: \n{model_response_text_list[1]}")
        print(f"\n\n###================== Average Log Probability List: \n{avg_log_prob_list}")


    elif model_id == 7:
        model_response_text_list, model_return_value_dict, chain_of_thought_text, avg_log_prob = deepseek_func(
            platform="DeepSeek_Official",                
            key_index=0, 
            model_name="deepseek-reasoner",   
            is_thinking=False, 
            input_role_play=INPUT_ROLE_PLAY, 
            input_problem_text=INPUT_PROBLEM_TEXT, 
            total_messages=[], 
            temperature=0.7, 
            max_length=1024, 
            is_output_prompt=True,
        )
        print(f"###================== Model Response Text: \n{model_response_text_list}")
        print(f"###================== Chain of Thought Text: \n{chain_of_thought_text}")
        print(f"###================== Average Log Probability: \n{avg_log_prob}")


# #####################################################################################################################🔖💡✅🟨
def general_llms_usage_preprocessing_func(key_index, input_role_play, input_problem_text, total_messages, is_output_prompt):
    if is_output_prompt and total_messages:
        print(f"\n\n###============================================= Total Messages:")
        pprint(total_messages)
    elif is_output_prompt:
        print(f"\n\n###============================================= Input Role Play: \n{input_role_play}")
        print(f"\n\n###============================================= Input Problem Text: \n{input_problem_text}")

    key_index = int(key_index) % 20

    if total_messages:
        used_messages = total_messages
    else:
        used_messages = [    {"role": "system", "content": input_role_play},
                             {"role": "user",   "content": input_problem_text}, ]
    
    return key_index, used_messages


# #####################################################################################################################🔖💡✅🟨
def general_llms_usage_postprocessing_func(model_name, is_output_prompt, model_response_text_list, avg_log_prob_list=[]):
    # --------------------------------------------
    if is_output_prompt:
        print(f"\n\n###============================================= Model: {model_name}, Response Text:")
        pprint(model_response_text_list)
        if avg_log_prob_list:
            print(f"\n\n###============================================= Model: {model_name}, Response Log Probs:\n{avg_log_prob_list}")


# #####################################################################################################################🔖💡✅🟨
def codellama_deepinfra_func(key_index, model_name="CodeGeneration2/CodeLlama-34b-Instruct-hf", input_role_play="You are an animal joke master.", input_problem_text='Tell a joke', total_messages=[], num_generations=1, temperature=1, is_output_prompt=False):

    # ---------------------------------------------------------------------------------------------------------  
    key_index, used_messages = general_llms_usage_preprocessing_func(key_index, input_role_play, input_problem_text, total_messages, is_output_prompt)

    client = OpenAI(
        api_key=CODELLAMA_API_KEYS[key_index],
        base_url="https://api.deepinfra.com/v1/openai",
    )

    model_return_value = client.chat.completions.create(
        model=model_name,
        messages=used_messages,
        max_tokens=1024,       # Or adjust as needed
        temperature=temperature,
        n=num_generations,                # How many chat completion choices to generate for each input message.
        # stream= True # or False
    )

    # --------------------------------------------------------------------------------------------
    model_response_text_list = []
    for index in range(num_generations):
        model_response_text_list.append(model_return_value.choices[index].message.content.strip())
    # model_response_text_list.append(model_return_value.choices[0].message.content)

    # -------------------------------------------------------------------------------------  
    general_llms_usage_postprocessing_func(model_name, is_output_prompt, model_response_text_list)

    sleep(1)

    return model_response_text_list


# ###############################################################################################################################################################🔖💡✅🟨
def gemini_official_func(key_index=0, model_name="gemini-1.5-pro", input_role_play="You are an animal joke master.", input_problem_text='Tell a joke', total_messages=[], num_generations=1, temperature=0.01, max_length="No Limit", thinking_budget=-404, return_log_probs=False, is_output_prompt=False):

    # ---------------------------------------------------------------------------------------------------------  
    key_index, used_messages = general_llms_usage_preprocessing_func(key_index, input_role_play, input_problem_text, total_messages, is_output_prompt)

    client = genai.Client(api_key=GEMINI_API_KEYS[key_index])

    safety_settings = [ {   "category": "HARM_CATEGORY_HARASSMENT", 
                            "threshold": "BLOCK_NONE"   }, 
                            {"category": "HARM_CATEGORY_HATE_SPEECH", 
                            "threshold": "BLOCK_NONE"  }, 
                            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", 
                            "threshold": "BLOCK_NONE"  }, 
                            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", 
                            "threshold": "BLOCK_NONE"  }, 
                            {"category": "HARM_CATEGORY_CIVIC_INTEGRITY", 
                            "threshold": "BLOCK_NONE"  }, ]

    context_list = [  types.Content(role="user",  parts=[types.Part.from_text(text="""Tell a joke""")]),
                        types.Content(role="model", parts=[types.Part.from_text(text="""A duck walks into a pharmacy and asks: 'Do you have lip balm?' The pharmacist says: 'Yes, what flavor do you want?' The duck says: 'Whatever, as long as it cures my cracked butt.'""")]),
                        types.Content(role="user",  parts=[types.Part.from_text(text="""Repeat the previous joke""")]),]

    
    # ------------------------------------------------------------------
    if (model_name == "gemini-2.5-flash") and (max_length == "No Limit") and (thinking_budget == -404):
        current_model_generation_config = types.GenerateContentConfig(
                            system_instruction=input_role_play, 
                            temperature=temperature,
                            candidate_count=num_generations,
                            )
    elif (model_name == "gemini-2.5-flash") and (max_length == "No Limit") and (thinking_budget != -404):
        current_model_generation_config = types.GenerateContentConfig(
                            thinking_config=types.ThinkingConfig(thinking_budget=thinking_budget),     # Disables thinking
                            system_instruction=input_role_play, 
                            temperature=temperature,
                            candidate_count=num_generations,
                            )
    elif (model_name == "gemini-2.5-flash") and (max_length == "No Limit") and (thinking_budget == -404):
        current_model_generation_config = types.GenerateContentConfig(
                            system_instruction=input_role_play, 
                            max_output_tokens=max_length,         # maxOutputTokens
                            temperature=temperature,
                            candidate_count=num_generations,
                            )
    elif return_log_probs:
        current_model_generation_config = types.GenerateContentConfig(
                            system_instruction=input_role_play, 
                            max_output_tokens=max_length,
                            temperature=temperature,
                            candidate_count=num_generations,
                            # safety_settings=safety_settings, 
                            responseLogprobs=return_log_probs,
                            # response_mime_type="text/plain",
                            )
    else:
        current_model_generation_config = types.GenerateContentConfig(
                            thinking_config=types.ThinkingConfig(thinking_budget=thinking_budget),     # Disables thinking
                            system_instruction=input_role_play, 
                            max_output_tokens=max_length,         # maxOutputTokens
                            temperature=temperature,
                            candidate_count=num_generations,
                            # safety_settings=safety_settings, 
                            # response_mime_type="text/plain",
                            )
        
    # ------------------------------------------------------------------
    model_return_value = client.models.generate_content(
                model=model_name,
                config=current_model_generation_config,
                contents=input_problem_text,       # context_list | input_problem_text
            )

    # --------------------------------------------------------------------------------------------
    if model_return_value.candidates[0].content.parts == None:
        print(f"### Gemini Official No Model Return Value")


    # --------------------------------------------------------------------------------------------
    """ Extract token-level log probabilities for each output from the candidates generated by multiple models, and calculate the average log probability for each output """
    model_response_text_list = [model_return_value.candidates[i].content.parts[0].text.strip() for i in range(num_generations)]

    # ----------------------------------------------------------------------------------------------------------
    """ Extract token-level log probabilities for each output from the candidates generated by multiple models (choices), calculate the average log probability for each output, 
    and then sort these candidates from high to low based on the average log probability. """
    avg_log_prob_list = []
    if return_log_probs:
        # log_prob_double_list = [[log_prob.log_probability for log_prob in single_model_gen_data.logprobs_result.chosen_candidates] for single_model_gen_data in model_return_value.candidates]
        avg_log_prob_list = [statistics.mean([log_prob.log_probability for log_prob in single_model_gen_data.logprobs_result.chosen_candidates]) for single_model_gen_data in model_return_value.candidates]
        # Keep 6 decimal places
        avg_log_prob_list = [round(num, 6) for num in avg_log_prob_list]

    # -------------------------------------------------------------------------------------  
    general_llms_usage_postprocessing_func(model_name, is_output_prompt, model_response_text_list, avg_log_prob_list)

    sleep(1)

    return model_response_text_list, avg_log_prob_list


# ###############################################################################################################################################################🔖💡✅🟨❌
def chatgpt_func(platform="OpenAI_Official", key_index=0, model_name="gpt-4o", input_role_play="You are an animal joke master.", input_problem_text='Tell a joke', total_messages=[], num_generations=1, temperature=0.01, max_length=1024, return_log_probs=False, is_output_prompt=False):

    # ---------------------------------------------------------------------------------------------------------  
    key_index, used_messages = general_llms_usage_preprocessing_func(key_index, input_role_play, input_problem_text, total_messages, is_output_prompt)

    if platform == "OpenAI_Official":
        client = OpenAI(api_key=CHATGPT_API_KEYS[key_index], )

    model_return_value = client.chat.completions.create(
        model=model_name,
        messages=used_messages,
        n=num_generations,                # How many chat completion choices to generate for each input message.
        temperature=temperature,       # GPT temperature [0, 2]
        logprobs=return_log_probs,      # Output, log probability
        max_completion_tokens=max_length,     # Maximum generated tokens
    ) 

    # ----------------------------------
    """ # 1) Convert the entire response to dict: 
        response_dict = model_return_value.to_dict()

        for choice in model_return_value.choices:
            # Convert Choice object to native dict
            choice_dict = choice.to_dict()  """

    # ----------------------------------------------------------------------------------------------------------
    """ Extract token-level log probabilities for each output from the candidates generated by multiple models (choices), calculate the average log probability for each output, 
        and then sort these candidates from high to low based on the average log probability. """
    log_prob_double_list = []
    avg_log_prob_list = []
    if return_log_probs == True:
        for single_model_gen_data in model_return_value.choices:
            single_log_prob_list = [log_prob.logprob for log_prob in single_model_gen_data.logprobs.content]
            log_prob_double_list.append(single_log_prob_list)
            avg_log_prob = statistics.mean(single_log_prob_list)
            avg_log_prob_list.append(avg_log_prob)
        prob_descending_index_order = sorted(range(len(avg_log_prob_list)), key=lambda x: avg_log_prob_list[x], reverse=True)


    # ----------------------------------------------------------------------------------------------------------
    """ Sort model_response_text_list based on the average log probability index, and return the sorted list. Prioritize returning "confident" replies. """
    model_response_text_list = []
    sorted_avg_log_prob_list = []
    if return_log_probs == True:
        for index in prob_descending_index_order:
            model_response_text_list.append(model_return_value.choices[index].message.content.strip())
            sorted_avg_log_prob_list.append(avg_log_prob_list[index])
    else:
        for index in range(num_generations):
            model_response_text_list.append(model_return_value.choices[index].message.content.strip())


    # --------------------------------------------------------------------------------------------
    """ Keep 6 decimal places """
    sorted_avg_log_prob_list = [round(num, 6) for num in sorted_avg_log_prob_list]
    for i, single_log_prob_list in enumerate(log_prob_double_list):
        log_prob_double_list[i] = [round(num, 6) for num in single_log_prob_list]

    # -------------------------------------------------------------------------------------  
    general_llms_usage_postprocessing_func(model_name, is_output_prompt, model_response_text_list, sorted_avg_log_prob_list)

    sleep(0.1)

    return model_response_text_list, sorted_avg_log_prob_list


# #################################################################################################################################################🔖💡✅🟨
def deepseek_func(platform="DeepSeek_Official", key_index=0, model_name='deepseek-chat', is_thinking=False, input_role_play="You are an animal joke master.", input_problem_text='Tell a joke', total_messages=[], temperature=0.01, max_length=1024, is_output_prompt=False):

    # ---------------------------------------------------------------------------------------------------------  
    key_index, used_messages = general_llms_usage_preprocessing_func(key_index, input_role_play, input_problem_text, total_messages, is_output_prompt)

    client = OpenAI( base_url = "https://api.deepseek.com/",
                     api_key = DEEPSEEK_API_KEYS[key_index]       )

    if (model_name=='deepseek-chat'):
        is_thinking=False
    elif (model_name=='deepseek-reasoner'):
        is_thinking=True

    if (platform == "DeepSeek_Official") and is_thinking == False:
        model_return_value = client.chat.completions.create(
            model=model_name,               # model='deepseek-reasoner'
            messages=used_messages,
            max_tokens=max_length,        # Integer between 1 and 8192, limits the maximum token count of model completion generation in a single request. Total length of input and output tokens is limited by model context length. Default is 4096.
            temperature=temperature,           # Sampling temperature, between 0 and 2. Higher values like 0.8 make output more random, lower values like 0.2 make it more focused and deterministic.
            logprobs=True,            # Whether to include log probabilities in the response. Default is false.
            # frequency_penalty = 0,  # Number between -2.0 and 2.0. If positive, new tokens are penalized based on their existing frequency in the text so far, decreasing the likelihood of the model repeating the same line verbatim.
            # presence_penalty = 0,   # Number between -2.0 and 2.0. If positive, new tokens are penalized based on whether they appear in the text so far, increasing the model's likelihood to talk about new topics.
            # stream=False,           # Stream output related option. Can only be set if stream parameter is true.
            # timeout=60  # Set timeout to 60 seconds
        )
    elif (platform == "DeepSeek_Official") and is_thinking == True:
        # Chain of thought reasoning model, does not support temperature parameter and others
        model_return_value = client.chat.completions.create(
            model=model_name, 
            messages=used_messages,
            max_tokens=max_length,        # Integer between 1 and 8192, limits the maximum token count of model completion generation in a single request. Total length of input and output tokens is limited by model context length. Default is 4096.
        )
    
    model_return_value_dict = model_return_value.to_dict()
    model_response_text_list = [model_return_value.choices[0].message.content.strip()]

    
    if is_thinking == False:
        chain_of_thought_text = ''
        avg_log_prob = statistics.mean([log_prob.logprob for log_prob in model_return_value.choices[0].logprobs.content])
    elif is_thinking == True:
        chain_of_thought_text = model_return_value.choices[0].message.reasoning_content.strip()
        avg_log_prob = ''


    # -------------------------------------------------------------------------------------  
    general_llms_usage_postprocessing_func(model_name, is_output_prompt, model_response_text_list)
    if is_thinking == True:
        print(f"###================================ Chain of Thought Text: {chain_of_thought_text}")
        
    sleep(0.1)

    return model_response_text_list, model_return_value_dict, chain_of_thought_text, avg_log_prob


# #####################################################################################################################🔖💡✅🟨
if __name__ == "__main__":
    main()