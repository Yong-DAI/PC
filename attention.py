
import torch
import torch.nn as nn
import torch.nn.functional as F
from wgnbase_embed import WGNbase

class PreT_Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.wgn_settings = {
            'image_size' : 224,
            'patch_size' : 16,
            'num_classes': 25,
            'dim' : 768,   ####1024
            'depth': 4,
            'heads' : 12,
            'mlp_dim' : 768*4,
            'dropout' : 0.0,
            'emb_dropout' : 0.0
        }
        
        self.wgn_module = WGNbase(   **self.wgn_settings    ).cuda()
        
    def forward(self, x, prompt):
        B, N, C = x.shape
        qkv = self.qkv(x)
        qkv_mh = qkv.reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        
        q, k, v = qkv_mh.unbind(0)   # make torchscript happy (cannot use tensor as tuple)
        q_f, k_f, v_f = qkv.reshape(B, N, 3, C).permute(2, 0, 1, 3).unbind(0)
        
        if prompt is not None:
            # prefix key, value
            prompt = prompt.permute(1, 0, 3, 2, 4).contiguous() # 2, B, num_heads, prompt_length, C // num_heads
            
            key_prefix = prompt[0] # B, num_heads, prompt_length, embed_dim // num_heads
            value_prefix = prompt[1] # B, num_heads, prompt_length, embed_dim // num_heads
            #############################
#             _, num_heads, prompt_length, miniembed_dim = key_prefix.shape
# # #            
#             K_p_mean =  key_prefix.permute(0, 2, 1, 3).reshape(B, prompt_length, -1)          
#             V_p_mean = value_prefix.permute(0, 2, 1, 3).reshape(B, prompt_length, -1)
            
            
#             k_prompts = self.wgn_module(k_f, K_p_mean)
#             v_prompts = self.wgn_module(v_f, V_p_mean)

#             key_prefix = k_prompts.reshape(B, prompt_length, num_heads, miniembed_dim).permute(0, 2, 1, 3)    # B, num_heads, prompt_length, embed_dim // num_heads
#             value_prefix = v_prompts.reshape(B, prompt_length, num_heads, miniembed_dim).permute(0, 2, 1, 3) # B, num_heads, prompt_length, embed_dim // num_heads
            ################################
            expected_shape = (B, self.num_heads, C // self.num_heads)
            
            assert (key_prefix.shape[0], key_prefix.shape[1], key_prefix.shape[3]) == expected_shape, f'key_prefix.shape: {key_prefix.shape} not match k.shape: {k.shape}'
            assert (value_prefix.shape[0], value_prefix.shape[1], value_prefix.shape[3]) == expected_shape, f'value_prefix.shape: {value_prefix.shape} not match v.shape: {v.shape}'

            k = torch.cat([key_prefix, k], dim=2)
            v = torch.cat([value_prefix, v], dim=2)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
#########################################################################   G and E weight good 85.93
#         if prompt is not None:
#             # prefix key, value
#             prompt = prompt.permute(1, 0, 3, 2, 4).contiguous() # 2, B, num_heads, prompt_length, C // num_heads
#             key_prefix = prompt[0] # B, num_heads, prompt_length, embed_dim // num_heads
#             value_prefix = prompt[1] # B, num_heads, prompt_length, embed_dim // num_heads
#             #############################
#             _, num_heads, prompt_length, miniembed_dim = key_prefix.shape
#             K_p_mean =  torch.mean(key_prefix.permute(0, 2, 1, 3).reshape(B, prompt_length, -1), dim = 0)
#             V_p_mean = torch.mean(value_prefix.permute(0, 2, 1, 3).reshape(B, prompt_length, -1), dim = 0)

#             K_p_mean =  key_prefix.permute(0, 2, 1, 3).reshape(B, prompt_length, -1)
#             V_p_mean = value_prefix.permute(0, 2, 1, 3).reshape(B, prompt_length, -1)
# #             if prompt_length ==5:
# #                 prompt_key =  self.prompt_key[idx].expand(B, -1).contiguous() 
# #             else: 
# #                 prompt_key =  self.prompt_key[0].expand(B, -1).contiguous()
#             K_p_mean_norm = self.line_norm(K_p_mean, dim=-1).reshape(B, prompt_length*3, 16,16) # N, C        ###20 768
#             V_p_mean_norm = self.line_norm(V_p_mean, dim=-1).reshape(B, prompt_length*3, 16,16)
# #             prompt_key_norm = self.line_norm(prompt_key, dim=-1).reshape(B, 3, 16,16)

#             x_embed_norm = self.line_norm(cls_features, dim=-1).reshape(B, 3, 16,16) # B, C

#             # similarity_t_K = torch.matmul(x_embed_norm, K_p_mean_norm.t())      ### B,  20

#             # Si_batch_tt = self.line_norm(similarity_t_K, dim=1).mean(dim=0)

#             fea_p_cat = torch.cat((x_embed_norm, K_p_mean_norm), 1)       ### B,63,16,16
#             fea_p_cat_V = torch.cat((x_embed_norm, V_p_mean_norm), 1) 
# #             fea_p_cat_PK = torch.cat((x_embed_norm, K_p_mean_norm,V_p_mean_norm), 1) 
#             if prompt_length ==20:
#                 weight1= self.proj1_a(fea_p_cat).reshape(B, -1)
#                 weight_t = self.proj2_a(weight1)#.squeeze()
#                 Si_batch_tt = self.sigmoid(weight_t)
# #                 Si_batch_t = torch.mean(Si_batch_tt, dim = 0,keepdim=False)#.expand(B, -1)

# #                 Si_batch_K_R = Si_batch_t # repeat(Si_batch_t, 'l  -> l c',   c= miniembed_dim)
#                 Si_batch_K_R =  repeat(Si_batch_tt.squeeze(), 'B  -> B h l c', h = num_heads, l = prompt_length,  c= miniembed_dim)
#                 #########
#                 weight1_V= self.proj1_a(fea_p_cat_V).reshape(B, -1)
#                 weight_t_V = self.proj2_a(weight1_V)#.squeeze()
#                 Si_batch_tt_V = self.sigmoid(weight_t_V)
# #                 Si_batch_t_Vm = torch.mean(Si_batch_tt_V, dim = 0,keepdim=False)#.expand(B, -1)

# #                 Si_batch_V_R = Si_batch_t_Vm# repeat(Si_batch_t_Vm, 'l  -> l c',  c= miniembed_dim)
#                 Si_batch_V_R =  repeat(Si_batch_tt_V.squeeze(), 'B  -> B h l c', h = num_heads, l = prompt_length,  c= miniembed_dim)

#             else:
# #             if prompt_length ==5:
#                 weight1= self.proj1_b(fea_p_cat).reshape(B, -1)
#                 weight_t = self.proj2_b(weight1)#.squeeze()
#                 Si_batch_tt = self.sigmoid(weight_t)
# #                 Si_batch_t = torch.mean(Si_batch_tt, dim = 0,keepdim=False)#.expand(B, -1)

# #                 Si_batch_K_R = Si_batch_t# repeat(Si_batch_t, 'l  -> l c',   c= miniembed_dim)
#                 Si_batch_K_R =  repeat(Si_batch_tt.squeeze(), 'B  -> B h l c', h = num_heads, l = prompt_length,  c= miniembed_dim)
#                 ##########

#                 weight1_V= self.proj1_b(fea_p_cat_V).reshape(B, -1)
#                 weight_t_V = self.proj2_b(weight1_V)#.squeeze()
#                 Si_batch_tt_V = self.sigmoid(weight_t_V)
# #                 Si_batch_t_Vm = torch.mean(Si_batch_tt_V, dim = 0,keepdim=False)#.expand(B, -1)

# #                 Si_batch_V_R = Si_batch_t_Vm # repeat(Si_batch_t_Vm, 'l  -> l c',  c= miniembed_dim)
#                 Si_batch_V_R =  repeat(Si_batch_tt_V.squeeze(), 'B  -> B h l c', h = num_heads, l = prompt_length,  c= miniembed_dim)
#                 ############
# #                 weight1_PK= self.proj1_c(fea_p_cat_PK).reshape(B, -1)
# #                 weight_t_PK = self.proj2_c(weight1_PK)#.squeeze()
# #                 Si_batch_tt_PK = self.sigmoid(weight_t_PK)
# #                 Si_batch_t_PKm = torch.mean(Si_batch_tt_PK, dim = 0,keepdim=False)#.expand(B, -1)

# #                 Si_batch_V_R = Si_batch_t_PKm   #repeat(Si_batch_t_PKm, 'l  -> l c',  c= miniembed_dim)
# #                 Si_batch_K_R = Si_batch_V_R
# #             else:
# #                 Si_batch_K_R = 1.0
# #                 Si_batch_V_R = 1.0
#             ############

#             # similarity_t_V = torch.matmul(x_embed_norm, V_p_mean_norm.t()) 

#             # Si_batch_tt_V = self.line_norm(similarity_t_V, dim=1).mean(dim=0)


#             # ###################################################################

#             expected_shape = (B, self.num_heads, C // self.num_heads)

#             assert (key_prefix.shape[0], key_prefix.shape[1], key_prefix.shape[3]) == expected_shape, f'key_prefix.shape: {key_prefix.shape} not match k.shape: {k.shape}'
#             assert (value_prefix.shape[0], value_prefix.shape[1], value_prefix.shape[3]) == expected_shape, f'value_prefix.shape: {value_prefix.shape} not match v.shape: {v.shape}'

#             key_prefix_W =key_prefix*Si_batch_K_R
#             value_prefix_W = value_prefix*Si_batch_V_R
#             k = torch.cat([key_prefix_W, k], dim=2)
#             v = torch.cat([value_prefix_W, v], dim=2)

# #             k = torch.cat([key_prefix, k], dim=2)
# #             v = torch.cat([value_prefix, v], dim=2)

#         attn = (q @ k.transpose(-2, -1)) * self.scale
#         attn = attn.softmax(dim=-1)
#         attn = self.attn_drop(attn)

#         x = (attn @ v).transpose(1, 2).reshape(B, N, C)
#         x = self.proj(x)
#         x = self.proj_drop(x)
#         return x
