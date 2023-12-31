import pinecone
import concurrent.futures
from torch import TensorType

class PineconeIndex():
    def __init__(self, conf="pinecone.conf"):
        with open(conf, 'r') as conf:
            lines = conf.readlines()

        api_key = lines[0].strip().strip('\n')
        enviro = lines[1].strip().strip('\n')
        index_name = lines[2].strip().strip('\n')
        
        pinecone.init(api_key=api_key,
                environment=enviro)

        self.index = pinecone.Index(index_name)

    def get_index(self):
        return self.index

    def upsert_vectors(self, vectors):
        upsert_response = self.index.upsert(
            vectors=vectors
        )
        return upsert_response

    def get_top_k(self, vectors, k):
        def process_vector(vector):
            # Function to process a single vector
            query_response = self.index.query(
                top_k=k,
                include_values=True,
                include_metadata=True,
                vector=vector,
            )
            ids, scores, values, metadata = [], [], [], []
            for match in query_response['matches']:
                ids.append(match['id'])
                scores.append(match['score'])
                values.append(match['values'])
                metadata.append(match['metadata'])
            return ids, scores, values, metadata

        all_ids, all_scores, all_values, all_metadata = [], [], [], []
        if type(vectors) == TensorType:
            vectors = vectors.cpu().numpy()
        if type(vectors) != list:
            vectors = vectors.tolist()
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(process_vector, vector) for vector in vectors]
            for future in concurrent.futures.as_completed(futures):
                ids, scores, values, metadata = future.result()
                all_ids.append(ids)
                all_scores.append(scores)
                all_values.append(values)
                all_metadata.append(metadata)

        return all_ids, all_scores, all_values
