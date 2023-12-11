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

#         x = self.original_model.forward_features(x_input)     
#         # print('x shape',x.shape)
#         x = x[:,0,:]
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

class PGNbase_3(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

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
        
        # self.input_embedding = vitembeding()  ### ResModel()

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )
        
        ##############prompt
        self.num_p = 25
        
        tl_vectors = torch.empty(
            256,
            768,
            # dtype=self.dtype,
            # device=self.device,
        )                                ####   [256,768]
        torch.nn.init.normal_(tl_vectors, std=0.02)
        self.tl_vectors = torch.nn.Parameter(tl_vectors)
        
        # self.tl_vectors = nn.Parameter(torch.randn((256, 768,)))
        # nn.init.uniform_(self.tl_vectors, -1, 1)
        
        self.acti_softmax =  nn.Softmax(dim=-1)
        self.acti_Sig = nn.Sigmoid()
               
#         self.input_frozen = FrozenVIT()
        
        self.pre_out = nn.Linear(768, self.num_p*2*256)
        
    def forward(self, x, maben=None):
        # print('img shape',img.shape)
        # x = self.to_patch_embedding(img)      ### 1,64 ,1024
        # with torch.no_grad():
        #     x = self.input_embedding(img)       ### b 1 768
        #     img_embedding = x.unsqueeze(1)
        x = x.unsqueeze(1)
        b,_,embed_dim = x.shape

        if maben ==None:
            prompt_tokens = repeat(self.tl_vectors, 'n d -> b n d', b = b)
        else:
            prompt_tokens = repeat(maben, 'n d -> b n d', b = b)
        # cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)     ## 1,1,1024
        
        # x = torch.cat((cls_tokens, x), dim=1)         b
        
        x = torch.cat((x, prompt_tokens), dim=1)          ###### b,257,7768
        # print(x.shape)
        # x += self.pos_embedding[:, :(n + 1)]      ##  1,65,1024

        # x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        
        corr_w = self.pre_out(x)
        
        split_logits = corr_w.reshape(
            b,
            self.num_p*2,   #### 16
           256          ###   256
        )
        mixture_coeffs = self.acti_softmax(
            split_logits
        )
        
        if maben ==None:
            pgn_prompts = torch.einsum(
            'bom,mv->bov',
            [mixture_coeffs, self.tl_vectors]
        ) 
        else:
            pgn_prompts = torch.einsum(
            'bom,mv->bov',
            [mixture_coeffs, maben]
        )
        return pgn_prompts#, img_embedding
        # return self.mlp_head(x)
