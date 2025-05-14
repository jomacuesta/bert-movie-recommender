from sentence_transformers import SentenceTransformer
from transformers import BertTokenizer, BertModel
from tqdm import tqdm
import torch
import numpy as np

class BertModelRecommender:
    def __init__(self, model_name='bert-base-uncased', pooling_strategy='mean'):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name)
        self.pooling_strategy = pooling_strategy
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

    def encode(self, text, batch_size = 32, **kwargs):
        if isinstance(text, str):
            text = [text]
        
        all_embeddings = []
        for i in tqdm(range(0, len(text), batch_size), desc="Encoding batches"):
            batch_text = text[i:i + batch_size]
            inputs = self.tokenizer(batch_text, return_tensors='pt', padding=True, truncation=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self.model(**inputs, **kwargs)
                pooled_output = self._pooling(outputs.last_hidden_state)
                all_embeddings.extend(pooled_output)

        return np.stack(all_embeddings) 
    
    def _pooling(self, outputs):
        """
        Pool the outputs of the BERT model to get a single vector representation.
        """
        # Mean pooling
        if self.pooling_strategy == 'mean':
            return outputs.mean(dim=1).squeeze().detach().cpu().numpy()
        # Max pooling
        elif self.pooling_strategy == 'max':
            return outputs.max(dim=1).values.squeeze().detach().cpu().numpy()
        # CLS token
        elif self.pooling_strategy == 'cls':
            return outputs[:, 0, :].squeeze().detach().cpu().numpy()
        else:
            raise ValueError(f"Unknown pooling strategy: {self.pooling_strategy}")

class SentenceTransformerRecommender:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)

    def encode(self, text, **kwargs):
        return self.model.encode(text, **kwargs)
    
    def similarity(self, input_embeddings, target_embeddings):
        """
        Calculate the cosine similarity between the target embeddings and all other embeddings.
        """
        return self.model.similarity(input_embeddings, target_embeddings)