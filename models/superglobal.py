import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from layers import GeM, Gemp, Rgem, Sgem, Relup
from .cvnet import GlobalNetwork

class SuperGlobalNetwork(GlobalNetwork):
    '''Don't include RelUP as only %1 increase in performance'''
    def __init__(self, reduction_dim=2048, resnet_depth=50):
        super().__init__(reduction_dim, resnet_depth)
        self.rgem = Rgem()
        self.gemp = Gemp()
        self.sgem = Sgem()

    def _forward_singlescale(self, x, gemp=True, rgem=True):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        if rgem:
            x = self.rgem(x)

        x = x.view(x.shape[0], x.shape[1], -1) 
        
        if gemp:
            x = self.gemp(x)

        x = x.view(x.size(0), -1)
        x = F.normalize(x, p=2, dim=-1)
        x = self.whitening(x)   

        return x

    def forward(self, x, scale=3, gemp=True, rgem=True, sgem=True):
        feature_list = []
        scales = [0.5, 0.7071, 1., 1.4142, 2.] if scale == 5 else ([0.7071, 1., 1.4142] if scale == 3 else [1.])
        
        for scl in scales:
            x_rescaled = TF.resize(x, [int(x.shape[-2]*scl), int(x.shape[-1]*scl)])
            features = self._forward_singlescale(x_rescaled, gemp, rgem)
            feature_list.append(features)
            
        if sgem:
            out_features = self.sgem(feature_list)
        else:
            out_features = torch.mean(torch.stack(feature_list, 0), 0)
            
        return out_features
       
class SuperGlobalRerank(torch.nn.Module):
    def __init__(self, vector_db, top_x=3, M=10, K=10, beta=2, device=None):
        self.vector_db = vector_db
        self.top_x = top_x
        self.M = M
        self.K = K
        self.beta = beta
        self.device = device if device else torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def foward(self, query_features):
        # Get the top M results
        top_m_ids, top_m_scores, top_m = self.vector_db.get_top_k(query_features, k=self.M)
        top_m = torch.tensor(top_m).to(self.device)
        
        # Construct g_de: extract top K, add the query descriptor, then maxpool
        top_k = top_m[:, :self.K-1]
        query_top_k = torch.concat((query_features.unsqueeze(1), top_k), dim=1) 
        query_top_k, _ = torch.max(query_top_k, dim=1)

        # We then want to find the top-K for each top-M to produce refined top-M (g_dr)
        top_k_m_scores, top_k_m = [], []
        for top_m_set in top_m:
            _, scores, values = self.vector_db.get_top_k(top_m_set, k=self.K-1)
            top_k_m_scores.append(scores)
            top_k_m.append(values)

        top_k_m_scores, top_k_m = torch.Tensor(top_k_m_scores).to(self.device), torch.Tensor(top_k_m).to(self.device)

        # For each top_m-top_k set, add the query descriptor to the set
        expanded_query_features = query_features.unsqueeze(1).unsqueeze(2).expand(-1, top_k_m.size(1), 1, -1)
        top_k_m = torch.cat((expanded_query_features, top_k_m), dim=2)

        # Add the highest possible score (1) for the image features in the cosine similarities
        ones = torch.ones((top_k_m.size(0), top_k_m.size(1), 1)).to(self.device)
        top_k_m_scores = torch.cat((ones, top_k_m_scores), dim=2)

        # Determine the weights (similarity * factor)
        weights = top_k_m_scores * self.beta

        # Compute the weighted sum of the top K descriptors
        weighted_top_k = top_k_m * weights.unsqueeze(-1)
        weighted_sum = weighted_top_k.sum(dim=2)

        # Normalizing factor (1 + sum of weights for each descriptor)
        normalizing_factor = 1 + weights.sum(dim=2, keepdim=True)

        # Compute refined descriptors - g_dr
        top_m_refined = weighted_sum / normalizing_factor

        # Normalize
        top_m_refined = F.normalize(top_m_refined, p=2, dim=-1)
        query_top_k = F.normalize(query_top_k, p=2, dim=-1)

        # Compute score for Set 1: (g_d, g_dr)
        score_1 = torch.einsum('ijk,ik->ij', top_m_refined, query_features)        
        
        # Set 2: (g_de, g_dr)
        score_2 = torch.einsum('ijk,ik->ij', top_m_refined, query_top_k)  

        # Final similarity score
        final_scores = (score_1 + score_2) / 2

        # Extract top x indices from the final scores
        top_x_scores, top_x_indices = torch.topk(final_scores, self.top_x, dim=1)

        return top_m_ids[top_x_indices], top_x_scores


class SuperGlobal(torch.nn.Module):
    def __init__(self, vector_db, reduction_dim=2048, resnet_depth=50, momentum=0.999):
        super(SuperGlobal, self).__init__()
        self.global_network = SuperGlobalNetwork(reduction_dim, resnet_depth)
        self.rerank_network = SuperGlobalRerank(vector_db)

    def forward(self, x):
        global_features = self.global_network(x)
        top_x_ids, top_x_scores = self.rerank_network(global_features)
        return top_x_ids, top_x_scores