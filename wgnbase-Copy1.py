import torch
from torch import nn
from torchvision import models
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from timm.models import create_model
# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)


# classes
# class ResModel(nn.Module):
#     def __init__(self):
#         super(ResModel, self).__init__()

#         self.premodel = models.resnet18(pretrained=True)
#         self.model = nn.Sequential(*list(self.premodel.children())[:-1])          ###[:-2]
#         out_chann = 512   ###  2048x4x4
#         print ("resnet18 model loaded")
#         self.compress = nn.Linear(out_chann, 768, bias=True)   ## False


#     def forward(self, x_input):
#         x = self.model(x_input)       #####  25 512 7 7
#         # print ('000:',x.shape)

#         x_comp = self.compress(x)      ###  25  2048

#         return x_comp

# class vitembeding(nn.Module):
#     def __init__(self):
#         super(vitembeding, self).__init__()

#         self.original_model = create_model(
#             'vit_base_patch16_224',
#             pretrained=True,
#             num_classes=1000,
#             drop_rate=0.0,
#             drop_path_rate=0.0,
#             drop_block_rate=None,
#         )


#     def forward(self, x_input):

#         for p in self.original_model.parameters():
#             p.requires_grad = False

#         x = self.original_model.forward_features(x_input)[:,0,:]       

#         return x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class WGNbase(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)
        self.num_classes = num_classes
        
        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),   ###  1,64,1024x3
            nn.Linear(patch_dim, dim),        ###  1,64,1024
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))     ###  1,65,1024
        
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
        
#         self.input_embedding = vitembeding()  ### ResModel()

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )
        ##########################

        self.acti_softmax =  nn.Softmax(dim=-1)
        self.acti_Sig = nn.Sigmoid()
                
        self.pre_out = nn.Linear(768, self.num_classes)
        
    def l2_normalize(self, x, dim=None, epsilon=1e-12):
        """Normalizes a given vector or matrix."""
        square_sum = torch.sum(x ** 2, dim=dim, keepdim=True)
        # print('square_sum shape:',square_sum.shape)                        ### [10,1] for prompt    [16,1] for x_embed_mean
        x_inv_norm = torch.rsqrt(torch.maximum(square_sum, torch.tensor(epsilon, device=x.device)))
        return x * x_inv_norm
    
    def trans_norm(self,x,dim=None):
        ttt_mean = torch.mean(x, dim=dim)
        ttt_var = torch.var(x, dim=dim)
        # # # print('ttmax:',ttt_max.shape)
        
        ttt_mean = repeat(ttt_mean, '...   -> ... n',  n = x.shape[dim])
        ttt_var = repeat(ttt_var, '...   -> ... n',  n = x.shape[dim])
        return (x-ttt_mean)/ttt_var 
    
    def line_norm(self,x,dim=None):
        ttt_max, ttindex_max = torch.max(x, dim=dim)
        ttt_min, ttindex_min = torch.min(x, dim=dim)
        # # # print('ttmax:',ttt_max.shape)
        
        ttt_max = repeat(ttt_max, '...   -> ... n',  n = x.shape[dim])
        ttt_min = repeat(ttt_min, '...   -> ... n',  n = x.shape[dim])
        return (x-ttt_min)/(ttt_max- ttt_min) 
    
#     def forward(self, embedding, prompt):
#         # x = self.to_patch_embedding(img)      ### 1,64 ,1024
#         # with torch.no_grad():
#         # x = self.input_embedding(img)
#         x_embed = embedding.unsqueeze(1)
#         b,_,wh = x_embed.shape
        
#         x_embed_norm_l = self.trans_norm(x_embed, dim=-1)
#         prompt_norm_l = self.trans_norm(prompt, dim=-1)
        
#         x = torch.cat((x_embed_norm_l, prompt_norm_l), dim=1)
        
# #         cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)     ## 1,1,1024
# #         x = torch.cat((cls_tokens, x), dim=1)          ###### 1,65 ,1024
#         # x += self.pos_embedding[:, :(n + 1)]      ##  1,65,1024
# #         x = self.dropout(x)

#         x = self.transformer(x)
#         x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]
#         x = self.to_latent(x)
        
#         weight_pre = self.pre_out(x)
#         weight_pre = self.acti_Sig(weight_pre)
        
#         ttt_max, ttindex_max = torch.max(weight_pre, dim=-1)
#         ttt_max_B = repeat(ttt_max, 'b -> b n', n =self.num_classes) 
# #         weight_pre = self.acti_Sig(weight_pre)
#         weight_pre_ = weight_pre/ttt_max_B
#         weight_pre_Si = repeat(weight_pre_, 'b n -> b n d', d = 768) 
        
#         weight_prompt= prompt*weight_pre_Si
        
#         return weight_prompt

    def forward(self, input_embed ,prompt_embed):
        
        x_embed = input_embed.unsqueeze(1)
        b,_,wh = x_embed.shape
        
        key = self.key_w(x_embed)
#         key = x_embed
        # print(key.size())
        
#         query = self.query_w(prompt_embed)
        query = prompt_embed
        # print(query.size())
        
        Q = query
        K = torch.transpose(key, 1, 2)
        V = prompt_embed
        
        scores = torch.matmul(Q, K)
        scores = (scores / math.sqrt(self.head_size)).squeeze()
        probs = torch.sigmoid(scores)
        
#         ttt_max, ttindex_max = torch.max(probs, dim=-1)
#         ttt_max_B = repeat(ttt_max, 'b -> b n', n =25) 
#         probs_ = probs/ttt_max_B

        probs = repeat(probs, 'b n -> b n d', d = 768) 
        prompt_embedding = torch.mul(V, probs)

        return prompt_embedding
