# Baselines



## Dependency

Python == 3.13.7

C++20

GCC 13.1.0

Linux

Run the following command in the root directory of this repo:

```sh
pip install -r requirements.txt
```




## Data

Please follow the instructions in the `processed_data/` folder to download the dataset for experiments.

## Usage of Single-Round Baselines

The source code of single-round baseline methods is in the `Single-Round/` folder.

For direct instruction:

```bash
cd Single-Round
bash direct.sh
```

For in-context learning:

```bash
cd Single-Round
bash icl.sh
```

For retrieval-augment generation:

```bash
cd Single-Round  
bash rag.sh
```

For chain-of-thought prompt:

```bash
cd Single-Round
bash cot.sh
```


## Usage of SBLLM Baseline

1. Download the processed dataset and test cases based on the instructions in the `processed_data/` folder. 

2. Our code relies on the service of OpenAI (for ChatGPT, GPT-4), Google (for Gemini), and DeepInfra (for CodeLLaMa), so you need first obtain their API keys and fill them in the `baselines/baselines.py` and `sbllm/evol_query.py` 

3. SBLLM acquires the initialization results based on the COT prompt, so you need first obtain the results of COT prompt by 

```bash
cd Single-Round
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
bash run.sh
```

SBLLM will then use default settings to optimize the code in the test set.

The default setting is set to `ns=3` and `iteration=4`. This setting is consistent with the paper. 



## Baseline results

Baseline results: The following are the results generated all the baselines on five different LLMs. 

|                | **Language** | **Generated Code (Includes CodeLlama-13b-Instruct-hf, CodeLlama-34b-Instruct-hf, Gemini-2.5-flash, GPT-3.5-turbo-0125 and DeepSeek-V3.2-Exp)** |
|:--------------:|:------------:|:-----------------------------------------------------------------------:|
| **PIE-C++**    | C++          | [PIE C++ Generated Code](https://drive.google.com/drive/folders/1FTZFwDGVn37NGD7vljsu7dk-dDmG_SN9?usp=sharing)                                                  |
| **PIE-Python** | Python       | [PIE Python Generated Code](https://drive.google.com/drive/folders/1w8_HcjD-_uhv5dlPvMHEULp5jE0gzV6C?usp=sharing)                                               |
| **PPIE**       | Python       | [PPIE Python Generated Code](https://drive.google.com/drive/folders/1rbiY_R1AI9cyGKQYm99tyHs_VJpohNM2?usp=sharing)                                              |



## Note

The baselines folder is mostly derived from the paper "Search-Based LLMs for Code Optimization", with its Github URL: https://github.com/shuzhenggao/sbllm
