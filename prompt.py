import argparse
import torch
import torch.nn as nn
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from pgnbase import PGNbase
from pgnbase_1 import PGNbase_1
from pgnbase_2 import PGNbase_2
# from pgnbase_3 import PGNbase_3
from wgnbase import WGNbase

# from wgnbase_embed import WGNbase
# from adapters.layer_prompt import MetaAdapterController,MetaLayersAdapterController

# def get_args_parser():
#     parser = argparse.ArgumentParser('DualPrompt CIFAR-100 training and evaluation configs', add_help=False)

#     parser.add_argument('--non_linearity', default="gelu_new", type=str)
#     parser.add_argument('--input_dim', default=768, type=int)

#     parser.add_argument('--task_embedding_dim', default=768, type=int)
#     parser.add_argument('--task_hidden_dim', default=128, type=int)
#     parser.add_argument('--projected_task_embedding_dim', default=768, type=int)
#     parser.add_argument('--device', default='cuda', type=str)
#     parser.add_argument('--unique_hyper_net_layer_norm', default=True, type=bool)
#     parser.add_argument('--reduction_factor', default=32, type=int)
#     return parser
def linear_layer(input_dim, output_dim, std=1e-2):
    """Generates a linear module and initializes it."""
    linear = nn.Linear(input_dim, output_dim)
#     init_linear_layer(linear, std=std)
    return linear

class EPrompt(nn.Module):
    def __init__(self, length=5, embed_dim=768, embedding_key='mean', prompt_init='uniform', prompt_pool=False, 
                 prompt_key=False, pool_size=None, top_k=None, batchwise_prompt=False, prompt_key_init='uniform',
                 num_layers=1, use_prefix_tune_for_e_prompt=False, num_heads=-1, same_key_value=False,):
        super().__init__()

        self.length = length
        self.prompt_pool = prompt_pool
        self.embedding_key = embedding_key
        self.prompt_init = prompt_init
        self.prompt_key = prompt_key
        self.pool_size = pool_size
        self.top_k = top_k
        self.batchwise_prompt = batchwise_prompt
        self.num_layers = num_layers
        self.use_prefix_tune_for_e_prompt = use_prefix_tune_for_e_prompt
        self.num_heads = num_heads
        self.same_key_value = same_key_value
        # self.lay_pgn_config = {
        #     "non_linearity": "gelu_new",
        #     "input_dim": 768,
        #     "task_embedding_dim": 768,
        #     "device": 0,
        #     "unique_hyper_net_layer_norm": True,
        #     "reduction_factor": 32
        # }
        
#         parser = argparse.ArgumentParser('DualPrompt configs', parents=[get_args_parser()])
#         args = parser.parse_args()
#         self.lay_pgn_module = MetaLayersAdapterController( args   )
        
        self.pgn_path = None      ### prompt generate network
        self.wgn_path = None      ### weight generate network
        self.pgn_settings = {
            'image_size' : 224,
            'patch_size' : 16,
            'num_classes': 25,
            'dim' : 768,   ####1024
            'depth' : 5,    ###5
            'heads' : 12,        ###12
            'mlp_dim' : 768*4,   ### 768*4
            'dropout' : 0.0,
            'emb_dropout' : 0.0
        }
#         self.wgn_settings = {
#             'image_size' : 224,
#             'patch_size' : 16,
#             'num_classes': 25,
#             'dim' : 768,   ####1024
#             'depth': 4,
#             'heads' : 12,
#             'mlp_dim' : 768*4,
#             'dropout' : 0.0,
#             'emb_dropout' : 0.0
#         }
        self.wgn_settings = {
            'num_classes': 25
        }
        self.pgn_module = PGNbase(    **self.pgn_settings    ).cuda()
        self.pgn_module_1 = PGNbase_1(    **self.pgn_settings    ).cuda()
        self.pgn_module_2 = PGNbase_2(    **self.pgn_settings    ).cuda()
        # self.pgn_module_3 = PGNbase_3(    **self.pgn_settings    ).cuda()
        self.wgn_module = WGNbase(   **self.wgn_settings    ).cuda()
        
#         if self.pgn_path:
#             self.load_pgn_module(self.pgn_path,self.pgn_settings)
#         else:
#             self.build_pgn_module(self.pgn_settings) 
            
#         if self.wgn_path:
#             self.load_wgn_module(self.wgn_path, self.wgn_settings)
#         else:
#             self.build_wgn_module(self.wgn_settings)
        
#         if self.prompt_pool:
#             # user prefix style
#             if self.use_prefix_tune_for_e_prompt:
#                 assert embed_dim % self.num_heads == 0
#                 if self.same_key_value:
#                     prompt_pool_shape = (self.num_layers, 1, self.pool_size, self.length, 
#                                         self.num_heads, embed_dim // self.num_heads)

#                     if prompt_init == 'zero':
#                         self.prompt = nn.Parameter(torch.zeros(prompt_pool_shape))
#                     elif prompt_init == 'uniform':
#                         self.prompt = nn.Parameter(torch.randn(prompt_pool_shape))
#                         nn.init.uniform_(self.prompt, -1, 1)
#                     self.prompt = self.prompt.repeat(1, 2, 1, 1, 1, 1)
#                 else:
#                     #######################################################
#                     prompt_pool_shape = (self.num_layers, 2, self.pool_size, self.length, 
#                                         self.num_heads, embed_dim // self.num_heads)
# #                     prompt_pool_shape = (pool_size, embed_dim)
#                     if prompt_init == 'zero':
#                         self.prompt = nn.Parameter(torch.zeros(prompt_pool_shape))
#                     elif prompt_init == 'uniform':
#                         self.prompt = nn.Parameter(torch.randn(prompt_pool_shape)) # num_layers, 2, pool_size, length, num_heads, embed_dim // num_heads
#                         nn.init.uniform_(self.prompt, -1, 1)
#             else:
#                 prompt_pool_shape=(self.num_layers, self.pool_size, self.length, embed_dim)
#                 if prompt_init == 'zero':
#                     self.prompt = nn.Parameter(torch.zeros(prompt_pool_shape))
#                 elif prompt_init == 'uniform':
#                     self.prompt = nn.Parameter(torch.randn(prompt_pool_shape))
#                     nn.init.uniform_(self.prompt, -1, 1)
                    
        # if using learnable prompt keys
#         eval_t = locals()
        if prompt_key:
      
            key_shape = (pool_size, embed_dim)
            if prompt_key_init == 'zero':
                self.prompt_key = nn.Parameter(torch.zeros(key_shape))
            elif prompt_key_init == 'uniform':
                self.prompt_key = nn.Parameter(torch.randn(key_shape))
                nn.init.uniform_(self.prompt_key, -1, 1)
#                 self.task_key_norm = torch.zeros(key_shape)
        else:
            # else use mean of prompt as key
            # only compatible with prompt, not prefix
            prompt_mean = torch.mean(self.prompt, dim=[0, 2])
            self.prompt_key = prompt_mean 
            
        self.maben = torch.empty(
            256,                        #### 
            768,
            # dtype=self.dtype,
#             device=self.device,
        ).cuda()                                ####   [256,768]
        torch.nn.init.normal_(self.maben, std=0.02)
        self.maben = torch.nn.Parameter(self.maben) 
#         self.tl_vectors = torch.nn.Parameter(self.maben) 
        
#         self.pgn_module_Gfc_1 = nn.Sequential(
#             linear_layer(768, 128),
#             nn.ReLU(),
#             linear_layer(128,768)) 
        
#         self.pgn_module_Gfc_2 = nn.Sequential(
#             linear_layer(768, 128),
#             nn.ReLU(),
#             linear_layer(128,768)) 
        
    def l2_normalize(self, x, dim=None, epsilon=1e-12):
        """Normalizes a given vector or matrix."""
        square_sum = torch.sum(x ** 2, dim=dim, keepdim=True)
        x_inv_norm = torch.rsqrt(torch.maximum(square_sum, torch.tensor(epsilon, device=x.device)))
        return x * x_inv_norm
    def load_pgn_module(self, pgn_path,pgn_settings):
        pgn_module  = PGNbase(     **pgn_settings    )

        pgn_module.load_state_dict(
            state_dict=torch.load(pgn_path)
        )
        self.pgn_module = pgn_module

    def build_pgn_module(self,pgn_settings):
        self.pgn_module = PGNbase(    **pgn_settings    ).cuda()

        
    def load_wgn_module(self, wgn_path,wgn_settings):
        wgn_module  = WGNbase(    **wgn_settings    )

        wgn_module.load_state_dict(
            state_dict=torch.load(wgn_path)
        )
        self.wgn_module = wgn_module

    def build_wgn_module(self,wgn_settings):
        self.wgn_module = WGNbase(   **wgn_settings    ).cuda()
    def forward(self, x_embed,task_id=-1,task_key_norm= None, prompt_mask=None, cls_features=None,trainable=False):
        out = dict()
        batch_size = x_embed.shape[0]
       
        if self.prompt_pool:
            if self.embedding_key == 'mean':
                x_embed_mean = torch.mean(x_embed, dim=1)
            elif self.embedding_key == 'max':
                x_embed_mean = torch.max(x_embed, dim=1)[0]
            elif self.embedding_key == 'mean_max':
                x_embed_mean = torch.max(x_embed, dim=1)[0] + 2 * torch.mean(x_embed, dim=1)
            elif self.embedding_key == 'cls':
                if cls_features is None:
                    x_embed_mean = torch.max(x_embed, dim=1)[0] # B, C
                else:
                    x_embed_mean = cls_features
            else:
                raise NotImplementedError("Not supported way of calculating embedding keys!")
                
#             task_key_norm =task_key_norm.cuda()
#             if  trainable:
#                 self.prompt_key_ = self.prompt_key
#                 prompt_key_norm = self.l2_normalize(self.prompt_key_, dim=-1) # Pool_size, C
# #                 print('prompt_key type', self.prompt_key.dtype)
                
#             else:
# #                 prompt_key_norm = torch.tensor(task_key_norm, dtype = torch.float32)
#                 prompt_key_norm = task_key_norm
#                 print('task_key_norm type', task_key_norm.dtype)
                
            prompt_key_norm = self.l2_normalize(self.prompt_key, dim=-1) # Pool_size, C
        
            x_embed_norm = self.l2_normalize(x_embed_mean, dim=-1) # B, C
            
            similarity = torch.matmul(prompt_key_norm, x_embed_norm.t()) # pool_size, B or Pool_size, #class, B
            similarity = similarity.t() # B, pool_size

            (similarity_top_k, idx) = torch.topk(similarity, k=self.top_k, dim=1) # B, top_k
            out['similarity'] = similarity
            out['idx_pred'] = idx
            
#             if self.batchwise_prompt and trainable:
#                 prompt_id, id_counts = torch.unique(idx, return_counts=True, sorted=True)
#                 # In jnp.unique, when the 'size' is specified and there are fewer than the indicated number of elements,
#                 # the remaining elements will be filled with 'fill_value', the default is the minimum value along the specified dimension.
#                 # Unless dimension is specified, this will be flattend if it is not already 1D.
#                 if prompt_id.shape[0] < self.pool_size:
#                     prompt_id = torch.cat([prompt_id, torch.full((self.pool_size - prompt_id.shape[0],), torch.min(idx.flatten()), device=prompt_id.device)])
#                     id_counts = torch.cat([id_counts, torch.full((self.pool_size - id_counts.shape[0],), 0, device=id_counts.device)])
#                 _, major_idx = torch.topk(id_counts, k=self.top_k) # top_k
#                 major_prompt_id = prompt_id[major_idx] # top_k
#                 # expand to batch
#                 idx = major_prompt_id.expand(x_embed.shape[0], -1).contiguous() # B, top_k
                
#                 out['idx_pred'] = idx

                # print('idx[0]',idx[0])
            # if prompt_mask is not None and trainable:
            if  trainable:
                idx = prompt_mask # B, top_k
                #print('when testing, this should not be used')    ## no
                
#             idx = prompt_mask
            out['idx_gt'] = prompt_mask
            out['prompt_idx'] = idx
            
            #############
            
            pgn_prompts_  = self.pgn_module(cls_features, self.maben)
#             out['prompt_l0'] = pgn_prompts_
#             batched_prompt = rearrange(pgn_prompts, 'b (l n e) (h d) -> b l n e h d',l=3,n=2, h = self.num_heads).permute(1, 0, 2,3,4,5)
            # print('shape:',pgn_prompts.shape)
            pgn_prompts_ = self.wgn_module(cls_features, pgn_prompts_)                            #### wgn
    #         pgn_prompts = self.wgn_module(x_embed, pgn_prompts)
            
            pgn_prompts=rearrange(pgn_prompts_.unsqueeze(0), 'l b (n e) (h d) -> l b n e h d', n = 2, h = self.num_heads)
            
            pgn_prompts_1  = self.pgn_module_1(cls_features, self.maben)
#             out['prompt_l1'] = pgn_prompts_1
            pgn_prompts_1 = self.wgn_module(cls_features, pgn_prompts_1)                            #### wgn
#             pgn_prompts_1 = self.pgn_module_Gfc_1(pgn_prompts_)
            pgn_prompts_1= rearrange(pgn_prompts_1.unsqueeze(0), 'l b (n e) (h d) -> l b n e h d', n = 2, h = self.num_heads)
            
            pgn_prompts_2  = self.pgn_module_2(cls_features, self.maben)
#             out['prompt_l2'] = pgn_prompts_2
            pgn_prompts_2 = self.wgn_module(cls_features, pgn_prompts_2)                           #### wgn
#             pgn_prompts_2 = self.pgn_module_Gfc_2(pgn_prompts_)
            pgn_prompts_2 = rearrange(pgn_prompts_2.unsqueeze(0), 'l b (n e) (h d) -> l b n e h d', n = 2, h = self.num_heads)
            
            # pgn_prompts_3  = self.pgn_module_3(cls_features, self.maben)
            # pgn_prompts_3 = self.wgn_module(cls_features, pgn_prompts_3)
#             #pgn_prompts_2 = self.pgn_module_Gfc_2(pgn_prompts_)
            # pgn_prompts_3 = rearrange(pgn_prompts_3.unsqueeze(0), 'l b (n e) (h d) -> l b n e h d', n = 2, h = self.num_heads)
        
            # batched_prompt  = torch.cat([pgn_prompts, pgn_prompts_1,pgn_prompts_2,pgn_prompts_3], dim=0)
            batched_prompt  = torch.cat([pgn_prompts, pgn_prompts_1,pgn_prompts_2], dim=0)
          
            # if self.use_prefix_tune_for_e_prompt:
            #     batched_prompt_raw = self.prompt[:,:,idx]  # num_layers,2, (B, top_k,) length, C
            #     num_layers, dual, batch_size, top_k, length, num_heads, heads_embed_dim = batched_prompt_raw.shape
                
            #     batched_prompt = batched_prompt_raw.reshape(
            #         num_layers, batch_size, dual, top_k * length, num_heads, heads_embed_dim
            #     )
            # else:
            #     batched_prompt_raw = self.prompt[:,idx]
            #     num_layers, batch_size, top_k, length, embed_dim = batched_prompt_raw.shape
            #     batched_prompt = batched_prompt_raw.reshape(
            #         num_layers, batch_size, top_k * length, embed_dim
            #     )

            batched_key_norm = prompt_key_norm[idx] # B, top_k, C

            out['selected_key'] = batched_key_norm
            out['prompt_key_norm'] = prompt_key_norm
            out['x_embed_norm'] = x_embed_norm
#             if task_id>0:
#                 task_key_norm = prompt_key_norm[:task_id+1]
#                 base_labels = torch.zeros(1,(task_id+1)).cuda()#.scatter(1, idx[0], 1) ##.expand(batch_size, -1)        ####[B, poolsize]
#                 base_labels = base_labels.index_fill(1, idx[0], 1)
#                 q_labels = torch.ones(batch_size, 1).cuda() 

#                 s = (q_labels @ base_labels > 0).float()     ####### [B, poolsize]
#                 # print('s shape:',s.shape)
#                 inner_product = x_embed_norm @ task_key_norm.t()      #####[B,poolsize]

#                 likelihood_loss = (1 + (-(inner_product.abs())).exp()).log() + inner_product.clamp(min=0) - s * inner_product
#                 # likelihood_loss = (1 -s)* inner_product - s * inner_product #+ inner_product.clamp(min=0)    ### 83.75
#                 lossDP = likelihood_loss.mean()                ### 0.6889
#             else:
#                 lossDP = 0.0
                
            # Put pull_constraint loss calculation inside
            lossDP =0.0
            weight_former = 1.0
            
#             if task_id> 0 and task_key_norm!= None:
# #                 if  trainable:
                    
# #                     x_xils = torch.ones(768)*int(task_id)
# #                     x_xils = x_xils.long().cuda()
# #                     y_xils = torch.arange(0,768).long().cuda()

# #                     new_value = prompt_key_norm[int(task_id)]
# #                     new_value = new_value.detach()

# #                     index = (
# #                             x_xils, 
# #                             y_xils,
# #                         )
# #                     task_key_norm.index_put_(index, new_value.cuda())
                
# #                     if task_id>0:

#                 former_idx = torch.arange(0,task_id).cuda()
#                 former_idx_batch =  former_idx.expand(x_embed.shape[0], -1).contiguous()
#                 former_key_norm = prompt_key_norm[former_idx_batch]

#                 x_embed_norm = x_embed_norm.unsqueeze(1) # B, 1, C
#                 sim = former_key_norm * x_embed_norm # B, task_id, C
#                 lossDP = torch.sum(sim) / x_embed.shape[0] # Scalar
#                 weight_former = 1.0/task_id

            x_embed_norm = x_embed_norm.unsqueeze(1) # B, 1, C
            sim = batched_key_norm * x_embed_norm # B, top_k, C
            reduce_sim = torch.sum(sim) / x_embed.shape[0] # Scalar
            
            out['reduce_sim'] =  reduce_sim# - weight_former*lossDP
#             out['reduce_sim'] = - lossDP
        # else:
        #     # user prefix style
        #     if self.use_prefix_tune_for_e_prompt:
        #         assert embed_dim % self.num_heads == 0
        #         if self.same_key_value:
        #             prompt_pool_shape = (self.num_layers, 1, self.length, 
        #                                 self.num_heads, embed_dim // self.num_heads)
        #             if self.prompt_init == 'zero':
        #                 self.prompt = nn.Parameter(torch.zeros(prompt_pool_shape))
        #             elif self.prompt_init == 'uniform':
        #                 self.prompt = nn.Parameter(torch.randn(prompt_pool_shape))
        #                 nn.init.uniform_(self.prompt, -1, 1)
        #             self.prompt = self.prompt.repeat(1, 2, 1, 1, 1)
        #         else:
        #             prompt_pool_shape = (self.num_layers, 2, self.length, 
        #                                 self.num_heads, embed_dim // self.num_heads)
        #             if self.prompt_init == 'zero':
        #                 self.prompt = nn.Parameter(torch.zeros(prompt_pool_shape))
        #             elif self.prompt_init == 'uniform':
        #                 self.prompt = nn.Parameter(torch.randn(prompt_pool_shape)) # num_layers, 2, length, num_heads, embed_dim // num_heads
        #                 nn.init.uniform_(self.prompt, -1, 1)
        #         batched_prompt = self.prompt.unsqueeze(0).expand(-1, x_embed.shape[0], -1, -1, -1)
        #     else:
        #         prompt_pool_shape = (self.num_layers, self.length, embed_dim)
        #         if self.prompt_init == 'zero':
        #             self.prompt = nn.Parameter(torch.zeros(prompt_pool_shape))
        #         elif self.prompt_init == 'uniform':
        #             self.prompt = nn.Parameter(torch.randn(prompt_pool_shape))
        #             nn.init.uniform_(self.prompt, -1, 1)
        #         batched_prompt = self.prompt.unsqueeze(0).expand(-1, x_embed.shape[0], -1, -1)
        
        out['batched_prompt'] = batched_prompt

        return out
