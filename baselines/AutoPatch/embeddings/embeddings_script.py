import json
import pandas as pd
import torch

# CodeBERT is based on the RoBERTa architecture, so the corresponding Tokenizer and Model are imported here.
from tqdm import tqdm
from transformers import RobertaTokenizer, RobertaModel

# ---------------------------------------------------------
# 1. Data Loading Stage
# ---------------------------------------------------------
# Open and read two files at the same time:
# one contains source code data, and the other contains CFG (Control Flow Graph) data.
with open(r'PIE_data\PIE_Cpp_008_base_table_reordered.csv', 'r', encoding='utf-8') as file1, \
     open(r'PIE_data\PIE_data.json', 'r', encoding='utf-8') as file2:
    df = pd.read_csv(file1)
    cfg_all_data = json.load(file2)  # Load the control flow graph dataset.

# Rename column names to make them consistent with the original JSON keys.
df = df.rename(columns={
    "712_idx": "id",
    "input": "source_code",
    "target": "optimized_code"
})

# Force the id column to be converted to string format.
# This is very important because the original JSON stores "id": "1" as a string.
# Keeping the same string type ensures successful matching with cfg_data_dict later.
df['id'] = df['id'].astype(str)

# Core step: convert the DataFrame into a list of dictionaries,
# perfectly simulating the original output format of json.load().
code_all_data = df.to_dict(orient='records')


# ---------------------------------------------------------
# 2. Model Initialization Stage
# ---------------------------------------------------------
# Initialize the CodeBERT tokenizer, which is used to split text/code into tokens that the model can understand.
tokenizer = RobertaTokenizer.from_pretrained(r'E:\Python_Params\CodeBERT-base')

# Initialize the pretrained CodeBERT model, which is used to extract high-dimensional features.
model = RobertaModel.from_pretrained(r'E:\Python_Params\CodeBERT-base')


# ---------------------------------------------------------
# 3. Core Function Definition
# ---------------------------------------------------------
def get_embedding(text):
    """
    Convert the input text, such as code or a CFG string, into its corresponding embedding representation.
    """
    # Convert the text into the input format required by the model:
    # return_tensors='pt' means returning PyTorch tensors.
    # padding=True and truncation=True ensure that input sequences have a unified length
    # and do not exceed the maximum length supported by the model.
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)

    # The torch.no_grad() context manager disables gradient computation.
    # Since the model is only used for inference here, that is, feature extraction,
    # backpropagation and weight updates are not required.
    # Disabling gradients can significantly reduce memory consumption and speed up computation.
    with torch.no_grad():
        outputs = model(**inputs)

    # Extract the feature representation:
    # 1. outputs.last_hidden_state obtains the hidden states from the last layer of the model,
    #    whose shape is usually [batch_size, sequence_length, hidden_size].
    # 2. .mean(dim=1) calculates the average over the sequence_length dimension, known as mean pooling.
    #    This merges the vectors of all tokens into a comprehensive vector representing the whole sentence,
    #    with the shape becoming [batch_size, hidden_size].
    # 3. .squeeze() removes all dimensions whose size is 1.
    #    Here, it mainly removes the batch_size dimension because only one text is processed at a time.
    # 4. .tolist() converts the PyTorch tensor into a regular Python list,
    #    making it easier to save into a CSV file later.
    return outputs.last_hidden_state.mean(dim=1).squeeze().tolist()


# ---------------------------------------------------------
# 4. Data Processing and Matching Stage
# ---------------------------------------------------------
# Initialize an empty list to store each row of data that will eventually be written to the CSV file.
rows = []

# To improve lookup efficiency, convert the cfg_all_data list into a dictionary.
# The key is the 'id' of each data entry, and the value is the corresponding 'source_cfg' text.
# In this way, the time complexity of later matching is reduced from O(N) to O(1).
cfg_data_dict = {str(entry['id']): entry['source_cfg'] for entry in cfg_all_data}

# Iterate over each record in the source code dataset.
for source_entry in tqdm(code_all_data, desc="Processing Source Code"):
    source_id = str(source_entry['id'])

    # Check whether the id of the current code also exists in the CFG data dictionary.
    # This ensures that the two datasets can be correctly aligned.
    if source_id in cfg_data_dict:
        # If the match is successful, extract the corresponding texts.
        source_code = source_entry['source_code']
        source_cfg = cfg_data_dict[source_id]

        # Call the function defined above to obtain the corresponding embedding representations.
        source_code_embedding = get_embedding(source_code)
        source_cfg_embedding = get_embedding(source_cfg)

        # Pack the matched ID and the two groups of feature vectors into a dictionary,
        # and append it to the rows list.
        rows.append({
            'id': source_id,
            'source_code_embeddings': source_code_embedding,
            'source_cfg_embeddings': source_cfg_embedding
        })


# ---------------------------------------------------------
# 5. Result Saving Stage
# ---------------------------------------------------------
# Use pandas to convert the list containing all results into a DataFrame.
df = pd.DataFrame(rows)

# Export the DataFrame to a CSV file.
# index=False means that the default row index of the DataFrame will not be written to the file.
df.to_csv(r'PIE_data\PIE_data_embeddings.csv', index=False)

# Print the completion message.
print("Embeddings saved!")