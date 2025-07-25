# Baselines



## Dependency

Python == 3.9.12

C++17

GCC 9.4.0

Linux

Run the following command in the root directory of this repo:

```sh
pip install -r requirements.txt
```



## Usage

1. Download the processed dataset and test cases based on the instructions in the `processed_data/` folder. 

2. Our code relies on the service of OpenAI (for ChatGPT, GPT-4), Google (for Gemini), and DeepInfra (for CodeLLaMa), so you need first obtain their API keys and fill them in the `baselines/baselines.py` and `sbllm/evol_query.py` 

3. SBLLM acquires the initialization results based on the COT prompt, so you need first obtain the results of COT prompt by 

```bash
cd baselines  
bash cot.sh
```
You can change the model name in the `cot.sh` to experiment on different models (i.e., chatgpt, gpt4, gemini, codellama)

4. Get the initailization solutions for SBLLM by processing the predictions of COT 

```bash
cd sbllm   
python initial.py --model_name model_name --lang lang
```

5. Modify the tree-sitter file path `TREE_SITTER_DIR` in `sbllm/merge.py` and run SBLLM with command

```bash
bash run.sh/run.sh
```

SBLLM will then use default settings to optimize the code in the test set.

The default setting is set to `ns=3` and `iteration=4`. This setting is consistent with the paper. 


## Data

Please follow the instructions in the `processed_data/` folder to download the dataset for experiments.

## Baselines

The source code of other baseline methods is in the `baselines/` folder.

For direct instruction:

```bash
cd baselines  
bash direct.sh
```

For in-context learning:

```bash
cd baselines  
bash icl.sh
```

For retrieval-augment generation:

```bash
cd baselines  
bash rag.sh
```

For chain-of-thought prompt:

```bash
cd baselines  
bash cot.sh
```

## Baseline results on PPIE

Baseline results on PPIE: The following are the [results generated](https://drive.google.com/file/d/1JH9f2-lBiAendFOXwChFkUDvFNku3vgB/view?usp=sharing) by SBLLM on four different LLMs. 


## SBLLM

The Baselines folder is derived from the paper "Search-Based LLMs for Code Optimization", with its Github URL: https://github.com/shuzhenggao/sbllm
