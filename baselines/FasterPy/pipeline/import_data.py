import sys
from pathlib import Path

# Get the absolute path of the current file and move up two levels to find the project root directory.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# Import the custom knowledge base module for later database interactions.
from knowledge.knowledge_base import KnowledgeBase

# Import the powerful data processing library pandas.
import pandas as pd

# Import the abstract syntax tree module.
# Here, literal_eval is mainly used to safely parse strings into Python objects, such as lists or vectors.
import ast


# Import the tqdm progress bar library.
from tqdm import tqdm

# Enable the tqdm extension for pandas so that DataFrame operations can display progress with progress_apply.
tqdm.pandas()


# #####################################################################################################################
# Import the pandarallel library for multiprocessing acceleration of pandas operations.
# from pandarallel import pandarallel

# Initialize the parallel computing environment and enable progress bars during parallel processing.
# pandarallel.initialize(progress_bar=True)


# Read a JSONL file from the specified path.
# lines=True indicates that each line is an independent JSON object.
print("Loading dataset...")
dataset = pd.read_json("OD-base.jsonl", lines=True)

# Print the dataset dimensions, i.e., number of rows and columns, to check whether the data is loaded correctly.
print(dataset.shape)

# Print all fields of the second row in the dataset, whose index is 1, to inspect a sample record.
print(dataset.loc[1, :])


# Data preprocessing: compute the number of lines of code in the "input" column.
# Use progress_apply to display the processing progress.
# The lambda function splits the code text by line and obtains the list length, i.e., the number of code lines.
print("Calculating source code line counts...")
dataset["source_code_length"] = dataset.progress_apply(
    lambda row: len(row["input"].splitlines()),
    axis=1
)


# Data preprocessing: parse vector data.
# The "vector" field read from JSON may be recognized as a string, such as "[0.1, 0.2, ...]".
# Use progress_apply to safely convert it into a real Python list object.
print("Parsing vector data...")
dataset["vector"] = dataset.progress_apply(
    lambda row: ast.literal_eval(row["vector"]),
    axis=1
)


# Initialize a knowledge base or vector database object named "CKB".
print("Initializing knowledge base...")
knowledge_base = KnowledgeBase(db_name="CKB")


# #####################################################################################################################
def insert_row(row):
    """
    Define a data insertion function.

    This function receives a single row from the DataFrame, assembles it into
    a dictionary, and inserts it into the knowledge base.
    """
    data = {
        "vector": row["vector"],                         # Parsed vector
        "source_code_length": row["source_code_length"], # Computed number of source code lines
        "summary": row["summary"],                       # Code summary or description
        "rate": row["rate"]                              # Score or weight
    }

    # Call the single-record insertion method of the knowledge base to write vector data into the database.
    knowledge_base.insert_single_with_vector(data)


# Apply the database insertion operation to each row of the DataFrame.
# Note: progress_apply is used here instead of parallel_apply.
# This is because database connections, such as SQLite or Milvus clients,
# are often not thread-safe or process-safe.
# Concurrent writes may cause database locks, connection crashes, or data loss.
# Therefore, preprocessing can use multiprocessing, while database insertion uses a single process.
dataset.progress_apply(insert_row, axis=1)