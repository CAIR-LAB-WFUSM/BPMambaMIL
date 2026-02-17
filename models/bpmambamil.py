"""
MambaMIL
"""
import os
import traceback
import torch
import torch.nn as nn
from mamba.mamba_ssm import SRMamba
from mamba.mamba_ssm import BiMamba
from mamba.mamba_ssm import Mamba
import torch.nn.functional as F




import numpy as np
def initialize_weights(module):
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()
        if isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba.mamba_ssm import SRMamba, BiMamba, Mamba


        
class AdaptivePrototypes(nn.Module):
    def __init__(self, num_prototypes, prototype_dim):
        super().__init__()
        self.prototypes = nn.Parameter(torch.randn(num_prototypes, prototype_dim))
        self.update_rate = 0.1

    def forward(self, features):
        similarities = F.cosine_similarity(features.unsqueeze(1), self.prototypes.unsqueeze(0), dim=-1)
        assignments = similarities.argmax(dim=1)
        for i in range(self.prototypes.size(0)):
            mask = (assignments == i)
            if mask.sum() > 0:
                self.prototypes.data[i] = (1 - self.update_rate) * self.prototypes.data[i] + \
                                          self.update_rate * features[mask].mean(dim=0)
        return self.prototypes

class PrototypeProcessor(nn.Module):
    def __init__(self, prototypes):
        super().__init__()
        self.prototypes = nn.Parameter(torch.cat(prototypes, dim=0), requires_grad=False)
    
    def forward(self):

        return self.prototypes



class MultiHeadCrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        assert self.head_dim * num_heads == dim, "dim must be divisible by num_heads"

        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.proj = nn.Linear(dim, dim)

        self.scale = self.head_dim ** -0.5

    def forward(self, x, prototype):
        B, L, _ = x.shape
        if prototype.dim() == 2:
            prototype = prototype.unsqueeze(0).expand(B, -1, -1)
        _, P, _ = prototype.shape

        # Linear projections and reshape
        q = self.query(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.key(prototype).view(B, P, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.value(prototype).view(B, P, self.num_heads, self.head_dim).transpose(1, 2)

        # Attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        # Combine heads
        out = (attn @ v).transpose(1, 2).contiguous().view(B, L, -1)
        out = self.proj(out)

        return out

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_dim, hidden_dim))
        for _ in range(num_layers - 2):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.layers.append(nn.Linear(hidden_dim, output_dim))
        self.activation = nn.ReLU()

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
        return self.layers[-1](x)
    
class PrototypePseudoBag(nn.Module):
    def __init__(self, prototype_dim=768, feature_dim=512, min_bag_size=3):
        
        super().__init__()
        self.min_bag_size = min_bag_size
        self.feature_projection = nn.Linear(prototype_dim, feature_dim)
        
        self.similarity_net = nn.Sequential(
            nn.Linear(feature_dim * 2, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    
    def forward(self, features, prototypes):

        batch_size, n_instances, feature_dim = features.shape
        device = features.device
        
      
        projected_protos = []
        for p in prototypes:
    
            p_flat = p.view(-1, p.size(-1)) 
            p_proj = self.feature_projection(p_flat)  # (1, 512)
            projected_protos.append(p_proj)
        
        enhanced_features_batch = []
        
        for b in range(batch_size):
            batch_features = features[b]  # (N, 512)
            proto_features = []
            
            for proj_proto in projected_protos:
           
                proto_expanded = proj_proto.expand(n_instances, -1)  # (N, 512)
                
     
                concat_features = torch.cat([batch_features, proto_expanded], dim=1)  # (N, 1024)
                similarity_scores = self.similarity_net(concat_features).squeeze()  # (N)
                
       
                k = min(max(self.min_bag_size, n_instances // 8), n_instances - 1)
           
                
         
                _, top_indices = torch.topk(similarity_scores, k)
                
          
                selected_features = batch_features[top_indices]  # (k, 512)
                proto_feature = selected_features.mean(0)  # (512)
                proto_features.append(proto_feature)
            
      
            proto_features = torch.stack(proto_features)  # (2, 512)
            enhanced_features_batch.append(proto_features)
        
    
        enhanced_features = torch.stack(enhanced_features_batch)  # (B, 2, 512)
        return enhanced_features

class MambaMIL(nn.Module):
    def __init__(self, in_dim, n_classes, dropout, act, survival=False, loss_type='class_balanced', samples_per_class=None, layer=2, rate=10, type="SRMamba",prototypes=None ):
        super(MambaMIL, self).__init__()
        self._fc1 = [nn.Linear(in_dim, 512)]
        if act.lower() == 'relu':
            self._fc1 += [nn.ReLU()]
        elif act.lower() == 'gelu':
            self._fc1 += [nn.GELU()]
        if dropout:
            self._fc1 += [nn.Dropout(dropout)]

        self._fc1 = nn.Sequential(*self._fc1)
        self.norm = nn.LayerNorm(512)
        self.layers = nn.ModuleList()
        self.survival = survival
        
        if type == "SRMamba":
            for _ in range(layer):
                self.layers.append(
                    nn.Sequential(
                        nn.LayerNorm(512),
                        SRMamba(
                            d_model=512,
                            d_state=16,  
                            d_conv=4,    
                            expand=2,
                        ),
                        )
                )
        elif type == "Mamba":
            for _ in range(layer):
                self.layers.append(
                    nn.Sequential(
                        nn.LayerNorm(512),
                        Mamba(
                            d_model=512,
                            d_state=16,  
                            d_conv=4,    
                            expand=2,
                        ),
                    )
                )
        elif type == "BiMamba":
            for _ in range(layer):
                self.layers.append(
                    nn.Sequential(
                        nn.LayerNorm(512),
                        BiMamba(
                            d_model=512,
                            d_state=16,  
                            d_conv=4,    
                            expand=2,
                        ),
                    )
                )
        else:
            raise NotImplementedError("Mamba [{}] is not implemented".format(type))

        self.n_classes = n_classes
        self.classifier = nn.Linear(512, self.n_classes)
        self.attention = nn.Sequential(
            nn.Linear(512, 128),
            nn.Tanh(),
            nn.Dropout(0.1),
            nn.Linear(128, 1)
        )


        # 添加对比学习的投影头
        self.projection_head = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            # nn.Dropout(0.25),
            nn.Linear(512, 1024)  
        )
        self.rate = rate
   
        self.temperature = nn.Parameter(torch.log(torch.tensor(1 / 0.07)))
        self.type = type

        if prototypes is not None:
            self.prototype_dim = prototypes[0].shape[-1]
            self.prototypes = nn.ParameterList([nn.Parameter(proto.clone()) for proto in prototypes])
        else:
            self.prototype_dim = 768 
            self.prototypes = nn.ParameterList([nn.Parameter(torch.randn(self.prototype_dim)) for _ in range(n_classes)])


        self.prototype_processor = PrototypeProcessor(prototypes) if prototypes is not None else None
        self.prototype_mapping = MLP(input_dim=1024, hidden_dim=640, output_dim=512, num_layers=3)  

        # self.prototypes = prototypes

        self.cross_attention = MultiHeadCrossAttention(512) if prototypes is not None else None
        self.prototype_layer = nn.Linear(512, 768)
        self.apply(initialize_weights)


        print("\nPrototypes initialized in MambaMIL:")
        if self.prototypes is not None:
            print(f"Number of prototypes: {len(self.prototypes)}")
            for i, proto in enumerate(self.prototypes):
                print(f"Prototype {i} shape: {proto.shape}")
        else:
            print("No prototypes provided.")

        if prototypes is not None:
            print(f"Initializing PrototypePseudoBag with {len(prototypes)} prototypes")
            self.pseudo_bag_generator = PrototypePseudoBag(
                prototype_dim=prototypes[0].shape[-1],  # 768
                feature_dim=512
            )
        else:
            self.pseudo_bag_generator = None




    def forward(self, x):

        if x.size(0) == 0 or x.size(1) == 0:
            raise ValueError(f"Invalid input dimensions: {x.shape}")
 
        if len(x.shape) == 2:
            x = x.unsqueeze(0)  
        
        h = self._fc1(x.float())
        

        

        if self.prototype_processor is not None and self.cross_attention is not None:
            try:
                prototypes = self.prototype_processor()  
                mapped_prototypes = self.prototype_mapping(prototypes)
                h = h + self.cross_attention(h, mapped_prototypes)

                if self.pseudo_bag_generator is not None:
            
                    proto_0 = prototypes[0:1]  # (1, 768)
                    proto_1 = prototypes[1:2]  # (1, 768)
                    
             
                    
            
                    pseudo_features = self.pseudo_bag_generator(h, [proto_0, proto_1])  # (B, 2, 512)
                    
                    
                    
              
                    h = torch.cat([h, pseudo_features], dim=1)  # (B, N+2, 512)
                    
                    
            except Exception as e:
                print(f"Error in forward pass: {e}")
                import traceback
                traceback.print_exc()
                # pseudo_features = None



        for layer in self.layers:
            h_ = h
            h = layer[0](h)
      
            
            h = layer[1](h, rate=self.rate)
      
            h = h + h_

 

        h = self.norm(h)
        
    
        if len(h.shape) == 2:
            h = h.unsqueeze(0)
        
        A = self.attention(h)
        A = torch.transpose(A, 1, 2)
        A = F.softmax(A, dim=-1)
        h = torch.bmm(A, h)
        h = h.squeeze(0)


        logits = self.classifier(h)
        Y_prob = F.softmax(logits, dim=1)
        Y_hat = torch.topk(logits, 1, dim=1)[1]
        
        contrast_features = self.projection_head(h)
        
        return logits, Y_prob, Y_hat, A, None,h,contrast_features




    def calculate_loss(self, logits, targets, features,contrast_features=None, pseudo_features=None,prototype_distances=None,pseudo_labels=None, alpha=0.5, temperature=0.5, threshold=0.9,lambda_proto=0.5):


        classification_loss = F.cross_entropy(logits, targets)


        contrastive_loss = self.contrastive_loss(contrast_features, targets)


        total_loss=0.5*classification_loss+0.5*contrastive_loss
  
        
        return total_loss
    

    
    def contrastive_loss(self, features, labels):
        device = features.device
        batch_size = features.size(0)

      
        similarity_matrix = F.cosine_similarity(features.unsqueeze(1), features.unsqueeze(0), dim=2)
        

        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)

        similarity_matrix = similarity_matrix / self.temperature.exp()

  
        logits_max, _ = torch.max(similarity_matrix, dim=1, keepdim=True)
        logits = similarity_matrix - logits_max.detach()
        

        exp_logits = torch.exp(logits)
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        if self.prototype_processor is not None:
            prototypes = self.prototype_processor()
            prototype_similarities = F.cosine_similarity(features.unsqueeze(1), prototypes.unsqueeze(0), dim=2)
            prototype_similarities = prototype_similarities / self.temperature.exp()
            
     
            k = min(self.n_classes, prototypes.size(0))  
            topk_similarities, topk_indices = torch.topk(prototype_similarities, k, dim=1)
            
    
            prototype_logits = topk_similarities - logits_max.detach()
            prototype_log_prob = prototype_logits - torch.log(exp_logits.sum(1, keepdim=True))
            
      
            soft_labels = F.softmax(topk_similarities, dim=1)
            mean_log_prob_pos_proto = (soft_labels * prototype_log_prob).sum(1) / soft_labels.sum(1)
            
     
            loss = - (mean_log_prob_pos + mean_log_prob_pos_proto) / 2
        else:
            loss = - mean_log_prob_pos
        
        return loss.mean()


    def relocate(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)

        
      
        self._fc1.to(device)
        self.norm.to(device)
        self.classifier.to(device)
        self.attention.to(device)
        self.prototype_layer.to(device)
        self.prototype_mapping.to(device)  

        for module in self.modules():
            if hasattr(module, 'to'):
                module.to(device)
        
        
