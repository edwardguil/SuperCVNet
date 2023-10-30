import pinecone

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
