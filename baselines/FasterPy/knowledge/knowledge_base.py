# This file implements vector knowledge base functionality.
from pymilvus import MilvusClient, CollectionSchema
from .code_embedder import CodeEmbedder
import numpy as np

# Define the default parameters of the Milvus database.
DEFAULT_DB_NAME = "milvus_demo"                # Default database name.
DEFAULT_COLLECTION_NAME = "demo_collection"    # Default collection name, similar to a table in a relational database.
DEFAULT_DIMENSION = 768                        # Default vector dimension, usually determined by the embedding model. For example, BERT-style models are usually 768-dimensional.

# Example of the expected input data format:
# Data format: {"vector": [0.1, 0.2, 0.3, 0.4, 0.5], "source_code": "source code string", "patch": "improvement patch string"}


def l1_normalize(data):
    """
    Compute L1 normalization.

    This scales the elements in the array so that the sum of their absolute
    values is 1. Note that this function is not directly used in the main
    logic of this script.
    """
    return data / np.sum(np.abs(data))


class KnowledgeBase:
    def __init__(
        self,
        db_name: str = DEFAULT_DB_NAME,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        dimension: int = DEFAULT_DIMENSION,
    ):
        """
        Initialize the knowledge base, connect to the Milvus database, and set
        the collection schema and index.
        """
        # Initialize the Milvus client and specify the local SQLite database file
        # for Milvus Lite mode.
        client = MilvusClient(f"{db_name}.db")

        # Check whether the specified collection exists.
        if client.has_collection(collection_name=collection_name):
            # If it exists, load the collection into memory for retrieval.
            client.load_collection(collection_name=collection_name)

            # Get the loading state.
            # The original commented-out code was used to check whether loading succeeded.
            load_state = client.get_load_state(collection_name=collection_name)["state"]
        else:
            # If the collection does not exist, define the schema and create it.
            from pymilvus import DataType, FieldSchema

            # Define the fields contained in the collection.
            fields = [
                # Primary key ID, set to auto-increment.
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),

                # Vector field used for semantic retrieval. The dimension must be specified.
                FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=dimension),

                # Source code length.
                # The original source_code field was commented out, possibly to save storage space
                # by storing only features and summaries.
                FieldSchema(
                    name="source_code_length",
                    dtype=DataType.INT64,
                    description="number of source code lines",
                ),

                # Summary description of the code modification. The maximum length is 300.
                FieldSchema(
                    name="summary",
                    dtype=DataType.VARCHAR,
                    description="summary of changes",
                    max_length=300,
                ),

                # Optimization improvement rate, such as the proportion by which runtime is reduced.
                FieldSchema(
                    name="rate",
                    dtype=DataType.FLOAT,
                    description="improvement rate",
                ),
            ]

            # Create the schema object.
            schema = CollectionSchema(fields=fields, auto_id=True, enable_dynamic_field=False)

            # Create the collection in Milvus.
            client.create_collection(
                collection_name=collection_name,
                schema=schema,
            )

            # Prepare and configure vector index parameters to accelerate retrieval.
            index_params = MilvusClient.prepare_index_params()
            index_params.add_index(
                field_name="vector",      # Field on which the index is built.
                metric_type="IP",         # Distance metric: IP means inner product. It is often used for cosine similarity when vectors are normalized.
                index_type="FLAT",        # Index type: FLAT means exact brute-force search, suitable for small datasets. HNSW can be used for large datasets.
                index_name="vector_index",
            )

            # Create the index.
            client.create_index(
                collection_name=collection_name,
                index_params=index_params,
                sync=True,  # Block until index creation is complete.
            )

        self.client = client

        # Initialize the code embedding model instance, which converts code strings into vectors.
        self.code_embedder = CodeEmbedder()

    def insert(self, data_list: list, collection_name: str = DEFAULT_COLLECTION_NAME):
        """
        Insert data in batch.

        This method automatically calls the embedding model to convert
        `source_code` into `vector`.
        """
        data_list = [
            {
                "vector": self.code_embedder(data["source_code"]),
                "source_code_length": data["source_code_length"],
                "summary": data["summary"],
                "rate": data["rate"],
            }
            for data in data_list
        ]
        self.client.insert(collection_name=collection_name, data=data_list)

    def insert_with_vector(self, data_list: list, collection_name: str = DEFAULT_COLLECTION_NAME):
        """
        Insert data whose vectors have already been computed in batch.
        """
        self.client.insert(collection_name=collection_name, data=data_list)

    def insert_single(self, data: dict, collection_name: str = DEFAULT_COLLECTION_NAME):
        """
        Insert one data record.

        This method automatically converts `source_code` into `vector`.
        """
        data = {
            "vector": self.code_embedder(data["source_code"]),
            "source_code_length": data["source_code_length"],
            "summary": data["summary"],
            "rate": data["rate"],
        }
        self.client.insert(collection_name=collection_name, data=data)

    def insert_single_with_vector(self, data: dict, collection_name: str = DEFAULT_COLLECTION_NAME):
        """
        Insert one data record that already contains a vector.
        """
        data = {
            "vector": data["vector"],
            "source_code_length": data["source_code_length"],
            "summary": data["summary"],
            "rate": data["rate"],
        }
        self.client.insert(collection_name=collection_name, data=data)

    def _filter(self, search_result: list, top_k: int = 2, distance_range: float = 0.2):
        """
        Custom result filtering and reranking logic.

        Goal: retrieve not only similar cases but also high-improvement-rate
        code cases within the similar range.
        """
        results = []

        for hits in search_result:
            # 1. Filter out exactly identical code, where distance == 1.0,
            # to avoid retrieving the query itself.
            hits = [hit for hit in hits if hit["distance"] != 1.0]

            # If the remaining results are fewer than top_k, keep all of them.
            if len(hits) <= top_k:
                results.append(hits)
                continue

            # 2. Sort by vector similarity, namely distance, in descending order.
            sorted_by_distance = sorted(hits, key=lambda x: x["distance"], reverse=True)

            # Always keep the most similar record.
            selected_results = [sorted_by_distance[0]]

            # Remove the most similar record and process the remaining data.
            sorted_by_distance = sorted_by_distance[1:]

            # Set an allowed lower bound for similarity.
            # Candidates are records whose similarity is within distance_range
            # below the second most similar result.
            minimum_distance = sorted_by_distance[0]["distance"] - distance_range
            sorted_by_distance = [
                hit for hit in sorted_by_distance
                if hit["distance"] >= minimum_distance
            ]

            # If the candidates within the specified distance range are not
            # enough to fill the remaining top_k slots, directly add them.
            if len(sorted_by_distance) < top_k - 1:
                selected_results.extend(sorted_by_distance)
                results.append(selected_results)
                continue

            # 3. If there are enough candidates within the distance range, sort
            # these close candidates by improvement rate in descending order.
            sorted_by_rate = sorted(
                sorted_by_distance,
                key=lambda x: x["entity"]["rate"],
                reverse=True,
            )

            # Add the records with the highest improvement rates to fill top_k.
            selected_results.extend(sorted_by_rate[:top_k - 1])
            results.append(selected_results)

        return results

    def search(
        self,
        query_codes: list = None,
        query_vector: list = None,
        rate_threshold: float = 0.1,
        similarity_threshold: float = 0.5,
        top_k: int = 2,
        search_limit: int = 10,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        output_fields: list = ["source_code_length", "summary", "rate"],
        metric_type: str = "IP",
        filter_function_enabled: bool = True,
    ):
        """
        Perform vector retrieval.

        This method supports retrieval by a list of source code strings or
        retrieval by a list of vectors directly.
        """
        # Check and process the query data.
        if query_codes:
            # The input is source code, so it must first be vectorized by the model.
            data = [self.code_embedder(query_code) for query_code in query_codes]
        elif query_vector:
            # Feature vectors are provided directly.
            data = query_vector
        else:
            raise Exception("No query code or vector was provided.")

        # Set the scalar filtering condition:
        # retrieve only data whose improvement rate is greater than the specified threshold.
        filter_expression = f"rate > {rate_threshold}"

        # Call the Milvus search interface.
        result = self.client.search(
            collection_name=collection_name,
            data=data,
            limit=search_limit,             # Upper bound of initially recalled results.
            output_fields=output_fields,    # Fields returned for each retrieved hit.
            filter=filter_expression,       # Apply the scalar filtering condition above.
            search_params={"metric_type": metric_type},  # Metric type. The default is IP.
        )

        # First filtering step: discard results whose vector similarity is lower
        # than similarity_threshold.
        result = [
            [hit for hit in hits if hit["distance"] >= similarity_threshold]
            for hits in result
        ]

        # Decide whether to apply the custom hybrid filtering defined above,
        # which considers both similarity and improvement rate.
        if filter_function_enabled:
            filtered_result = self._filter(result, top_k=top_k)
            return filtered_result
        else:
            return result

    def drop_collection(self, collection_name: str = DEFAULT_COLLECTION_NAME):
        """
        Drop the specified collection, removing all data and deleting the schema.
        """
        self.client.drop_collection(collection_name=collection_name)


if __name__ == "__main__":
    # The following section is a test module.

    code1 = """
        #include <iostream>

    void reverseString(std::string &s) {
        int left = 0, right = s.length() - 1;
        while (left < right) {
            std::swap(s[left], s[right]);
            left++;
            right--;
        }
    }

    int main() {
        std::string str = "hello";
        reverseString(str);
        std::cout << str << std::endl;  // Output: "olleh"
        return 0;
    }
    """

    code2 = """
    int useless(){
    // comments
    int a = 1;
    int b = 2;
    int c = a + b;
    printf("c: %d\n", c);
    return c;
}
    """

    code3 = """
D, G = list(map(int, input().split()))
P = []
num_p = 0
for i in range(1, D + 1):
    p, c = list(map(int, input().split()))
    num_p += p
    for j in range(1, p + 1):
        P += [(j, i * 100 * j + c * (j == p))]
dp = [0] * (num_p + 1)
for k, pt in P:
    if k == 1:
        t_dp = dp[:]
    for cur in range(1, num_p + 1):
        if cur >= k:
            dp[cur] = max(dp[cur], t_dp[cur - k] + pt)
for i in range(num_p + 1):
    if dp[i] >= G:
        print(i)
        break
    """

    # Instantiate the knowledge base object and specify the database name as CKB.
    knowledge_base = KnowledgeBase(db_name="CKB")

    # Commented-out test functions.
    # knowledge_base.drop_collection()  # Test dropping the collection.
    # data_list = (
    #     {"source_code": code1, "patch": "patch1", "rate": 0.7},
    #     {"source_code": code2, "patch": "patch2", "rate": 0.3},
    # )
    # knowledge_base.insert(data_list=data_list)  # Test data insertion.

    import time

    start_time = time.time()

    # Simulate concurrent or large-scale retrieval testing:
    # repeat code1, code2, and code3 100 times, producing 300 queries in total,
    # and send them to the knowledge base together.
    result = knowledge_base.search([code1, code2, code3] * 100)

    # Print the total time consumed by the 300 retrieval operations.
    print(time.time() - start_time)

    # Commented-out output function.
    # Print detailed retrieval hit results.
    # for hits in result:
    #     for hit in hits:
    #         print(hit, "\n")