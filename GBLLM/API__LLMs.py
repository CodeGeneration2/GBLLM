from time import sleep
import statistics
from openai import OpenAI
from google import genai
from google.genai import types

# API key lists
deepseek_api_keys = []
chatgpt_api_keys = []
gemini_api_keys = []
codellama_api_keys = []

# Default role and prompt
input_role = "You are a master of animal jokes."
input_prompt = "Tell me a joke."

def main(model_number=1):
    if model_number == 1:
        responses, thoughts, logprobs_nested, avg_logprobs = call_openai_chatgpt(
            api_key=chatgpt_api_keys[0],
            model_name="gpt-4-1106-preview",
            input_role=input_role,
            input_prompt=input_prompt,
            num_responses=3,
            temperature=0.7,
            return_logprobs=True
        )
        print(f"\n\n### responses: \n{responses}")
        for i, resp in enumerate(responses):
            print(f"\n\n### responses[{i}]: \n{resp}")
        print(f"\n\n### chain of thought: \n{thoughts}")
        print(f"\n\n### nested logprobs: \n{logprobs_nested}")
        print(f"\n\n### average logprobs: \n{avg_logprobs}")

    elif model_number == 2:
        responses, avg_logprobs = call_gemini(
            api_key=gemini_api_keys[0],
            model_name="gemini-1.5-pro",
            input_role=input_role,
            input_prompt=input_prompt,
            num_responses=3,
            temperature=0.7
        )
        print(f"\n\n### responses: \n{responses}")
        for i, resp in enumerate(responses):
            print(f"\n\n### responses[{i}]: \n{resp}")
        print(f"\n\n### average logprobs: \n{avg_logprobs}")

    elif model_number == 3:
        responses = call_code_llama(
            api_key=codellama_api_keys[0],
            model_name="CodeGeneration2/CodeLlama-34b-Instruct-hf",
            input_role=input_role,
            input_prompt=input_prompt,
            num_responses=3,
            temperature=0.7
        )
        print(f"\n\n### responses: \n{responses}")
        for i, resp in enumerate(responses):
            print(f"\n\n### responses[{i}]: \n{resp}")

def call_openai_chatgpt(
    api_key,
    model_name="gpt-4o",
    input_role="You are a master of animal jokes.",
    input_prompt="Tell me a joke.",
    num_responses=1,
    temperature=1,
    return_logprobs=False
):
    client = OpenAI(api_key=api_key)

    messages = [
        {"role": "system", "content": input_role},
        {"role": "user", "content": input_prompt},
    ]

    result = client.chat.completions.create(
        model=model_name,
        messages=messages,
        n=num_responses,                # number of completions to generate
        temperature=temperature,
        logprobs=return_logprobs,       # whether to return token logprobs
        max_completion_tokens=2048,
    )

    # Gather log probabilities if requested
    nested_logprobs = []
    avg_logprobs = []
    if return_logprobs:
        for choice in result.choices:
            probs = [lp.logprob for lp in choice.logprobs.content]
            nested_logprobs.append(probs)
            avg_logprobs.append(statistics.mean(probs))

    # Sort responses by descending average log probability
    sorted_indices = sorted(
        range(len(avg_logprobs)),
        key=lambda i: avg_logprobs[i],
        reverse=True
    ) if return_logprobs else list(range(len(result.choices)))

    # Collect sorted outputs
    sorted_responses = []
    chain_of_thought = []
    sorted_nested = []
    sorted_avg = []

    for idx in sorted_indices:
        choice = result.choices[idx]
        sorted_responses.append(choice.message.content)
        # include chain-of-thought if available
        if model_name == "deepseek-reasoner":
            chain_of_thought.append(choice.message.reasoning_content)
        if return_logprobs:
            sorted_nested.append(nested_logprobs[idx])
            sorted_avg.append(round(avg_logprobs[idx], 6))

    # Round nested logprobs
    sorted_nested = [
        [round(lp, 6) for lp in seq] for seq in sorted_nested
    ]

    sleep(1)
    return sorted_responses, chain_of_thought, sorted_nested, sorted_avg

def call_gemini(
    api_key,
    model_name="gemini-1.5-pro",
    input_role="You are a master of animal jokes.",
    input_prompt="Tell me a joke.",
    num_responses=1,
    temperature=1,
    allow_retry=True
):
    client = genai.Client(api_key=api_key)

    result = client.models.generate_content(
        model=model_name,
        config=types.GenerateContentConfig(
            system_instruction=input_role,
            max_output_tokens=2048,
            temperature=temperature,
            candidate_count=num_responses,
        ),
        contents=input_prompt,
    )

    responses = []
    avg_logprobs = []
    for cand in result.candidates:
        if cand.content and cand.content.parts:
            text = cand.content.parts[0].text.strip()
            responses.append(text)
            avg_logprobs.append(cand.avg_logprobs)

    # Retry if not enough responses
    if allow_retry and len(responses) < num_responses:
        current_temp = temperature
        for attempt in range(15):
            extra_resps, extra_avgs = call_gemini(
                api_key, model_name, input_role, input_prompt,
                num_responses, current_temp, allow_retry=False
            )
            responses.extend(extra_resps)
            avg_logprobs.extend(extra_avgs)
            if len(responses) >= num_responses:
                break
            if attempt == 5:
                current_temp = 2
        responses = responses[:num_responses]
        avg_logprobs = avg_logprobs[:num_responses]
        assert len(responses) >= num_responses, (
            f"Error: not enough Gemini responses generated: {responses}"
        )

    sleep(2)
    return responses, avg_logprobs

def call_code_llama(
    api_key,
    model_name="CodeGeneration2/CodeLlama-34b-Instruct-hf",
    input_role="You are a master of animal jokes.",
    input_prompt="Tell me a joke.",
    num_responses=1,
    temperature=1
):
    client = OpenAI(
        api_key=api_key,
        base_url="https://api.deepinfra.com/v1/openai"
    )

    result = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": input_role},
            {"role": "user", "content": input_prompt},
        ],
        max_tokens=1024,
        temperature=temperature,
        n=num_responses,
    )

    responses = [
        choice.message.content for choice in result.choices
    ]

    sleep(1)
    return responses

if __name__ == "__main__":
    main()
