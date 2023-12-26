import math
from functools import partial
import scipy.io as sio
import torch
import torch.nn as nn
from torch import _assert
from timm.models.vision_transformer import Block
from timm.models.layers import to_2tuple, DropPath
# from timm.models.vision_transformer import PatchEmbed, Block
from pos_embed import get_2d_sincos_pos_embed

class PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding
    """
    def __init__(self, img_size=(224, 224), patch_size=16, in_chans=3, embed_dim=768, norm_layer=None, flatten=True):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
         
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        _assert(H == self.img_size[0], f"Input image height ({H}) doesn't match model ({self.img_size[0]}).")
        _assert(W == self.img_size[1], f"Input image width ({W}) doesn't match model ({self.img_size[1]}).")
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb

class MaskedAutoencoderViT(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, in_chans_LIDAR = 1,hid_chans = 32,hid_chans_LIDAR = 32,
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., drop_rate=0.,attn_drop_rate=0., drop_path_rate=0., 
                 norm_layer=nn.LayerNorm, norm_pix_loss=False,
                 cls_hidden_mlp=256, nb_classes=1000, global_pool=False,
                 mlp_depth=2):
        super().__init__()
        # --------------------------------------------------------------------------
        #HSI
        # MAE dimensionality reduction/expansion specifics
        self.dimen_redu = nn.Sequential(
            nn.Conv2d(in_chans, hid_chans, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(hid_chans),
            nn.ReLU(),
            nn.Conv2d(hid_chans, hid_chans, 1, 1, 0, bias=True),
            nn.BatchNorm2d(hid_chans),
            nn.ReLU(),
            )

        self.t_embedder = TimestepEmbedder(embed_dim)
        self.dimen_expa = nn.Conv2d(hid_chans, in_chans, kernel_size=1, stride=1, padding=0, bias=True)
        self.h = img_size[0]// patch_size
        self.w = img_size[1]// patch_size
        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_embed = PatchEmbed(img_size, patch_size, hid_chans , embed_dim)
        num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)  # fixed sin-cos embedding
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, drop=drop_rate,attn_drop=attn_drop_rate, drop_path=drop_path_rate, qkv_bias=True, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        self.projection_en_mask = nn.Linear(embed_dim, embed_dim)
        self.projection_en_visible = nn.Linear(embed_dim, embed_dim)
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding
        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(decoder_depth)])
        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size**2 * hid_chans, bias=True) # decoder to patch
        self.projection_de = nn.Linear(decoder_embed_dim, decoder_embed_dim)
        if cls_hidden_mlp == 0:
            self.cls_head = nn.Linear(embed_dim, nb_classes)
        else:
            assert mlp_depth in [2], "mlp depth should be 2"
            if mlp_depth == 2:
                self.cls_head = nn.Sequential(
                    nn.Linear(embed_dim*2, cls_hidden_mlp),
                    nn.BatchNorm1d(cls_hidden_mlp),
                    nn.ReLU(inplace=True),
                    nn.Linear(cls_hidden_mlp, embed_dim),
                    nn.BatchNorm1d(embed_dim),
                    nn.ReLU(inplace=True),
                    nn.Linear(embed_dim, nb_classes),
                )
        # --------------------------------------------------------------------------
        self.global_pool = global_pool
        self.norm_pix_loss = norm_pix_loss

        # --------------------------------------------------------------------------
        # LIDAR
        # MAE dimensionality reduction/expansion specifics
        self.dimen_redu_LIDAR = nn.Sequential(
            nn.Conv2d(in_chans_LIDAR, hid_chans_LIDAR, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(hid_chans_LIDAR),
            nn.ReLU(),

            nn.Conv2d(hid_chans_LIDAR, hid_chans_LIDAR, 1, 1, 0, bias=True),
            nn.BatchNorm2d(hid_chans_LIDAR),
            nn.ReLU(),
        )
        self.t_embedder_LIDAR = TimestepEmbedder(embed_dim)
        self.dimen_expa_LIDAR = nn.Conv2d(hid_chans_LIDAR, in_chans_LIDAR, kernel_size=1, stride=1, padding=0, bias=True)
        self.h = img_size[0] // patch_size
        self.w = img_size[1] // patch_size
        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_embed_LIDAR = PatchEmbed(img_size, patch_size, hid_chans_LIDAR, embed_dim)
        num_patches = self.patch_embed_LIDAR.num_patches

        self.cls_token_LIDAR = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed_LIDAR = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim),
                                      requires_grad=False)  # fixed sin-cos embedding

        self.blocks_LIDAR = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=drop_path_rate,
                  qkv_bias=True, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm_LIDAR = norm_layer(embed_dim)
        self.projection_en_mask_LIDAR = nn.Linear(embed_dim, embed_dim)
        self.projection_en_visible_LIDAR = nn.Linear(embed_dim, embed_dim)
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed_LIDAR = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token_LIDAR = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed_LIDAR = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim),
                                              requires_grad=False)  # fixed sin-cos embedding

        self.decoder_blocks_LIDAR = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(decoder_depth)])

        self.decoder_norm_LIDAR = norm_layer(decoder_embed_dim)
        self.decoder_pred_LIDAR = nn.Linear(decoder_embed_dim, patch_size ** 2 * hid_chans, bias=True)  # decoder to patch
        self.projection_de_LIDAR = nn.Linear(decoder_embed_dim, decoder_embed_dim)
                # --------------------------------------------------------------------------
        self.global_pool_LIDAR = global_pool
        self.norm_pix_loss_LIDAR = norm_pix_loss


        self.initialize_weights()

    def initialize_weights(self):
        # initialization`

        # initialize (and freeze) pos_embed by sin-cos embedding
        #HSI
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)
        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        #LIDAR
        pos_embed_LIDAR = get_2d_sincos_pos_embed(self.pos_embed_LIDAR.shape[-1], int(self.patch_embed_LIDAR.num_patches**.5), cls_token=True)
        self.pos_embed_LIDAR.data.copy_(torch.from_numpy(pos_embed_LIDAR).float().unsqueeze(0))

        decoder_pos_embed_LIDAR = get_2d_sincos_pos_embed(self.decoder_pos_embed_LIDAR.shape[-1], int(self.patch_embed_LIDAR.num_patches**.5), cls_token=True)
        self.decoder_pos_embed_LIDAR.data.copy_(torch.from_numpy(decoder_pos_embed_LIDAR).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w_LIDAR = self.patch_embed_LIDAR.proj.weight.data
        torch.nn.init.xavier_uniform_(w_LIDAR.view([w_LIDAR.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token_LIDAR, std=.02)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder_LIDAR.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder_LIDAR.mlp[2].weight, std=0.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs, imgs_LIDAR):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_embed.patch_size[0]
        p_LIDAR = self.patch_embed_LIDAR.patch_size[0]
        # assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = imgs.shape[2] // p
        w = imgs.shape[3] // p

        x = imgs.reshape(shape=(imgs.shape[0], imgs.shape[1], h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * imgs.shape[1]))

        x_LIDAR = imgs_LIDAR.reshape(shape=(imgs_LIDAR.shape[0], imgs_LIDAR.shape[1], h, p_LIDAR, w, p_LIDAR))
        x_LIDAR = torch.einsum('nchpwq->nhwpqc', x_LIDAR)
        x_LIDAR = x_LIDAR.reshape(shape=(imgs_LIDAR.shape[0], h * w, p_LIDAR**2 * imgs_LIDAR.shape[1]))
        return x, x_LIDAR

    def unpatchify(self, x, x_LIDAR):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_embed.patch_size[0]
        p_LIDAR = self.patch_embed_LIDAR.patch_size[0]
        h = self.h
        w = self.w
        assert h * w == x.shape[1]
        assert h * w == x_LIDAR.shape[1]

        hid_chans = int(x.shape[2]/(p**2))
        hid_chans_LIDAR = int(x_LIDAR.shape[2]/(p_LIDAR**2))

        x = x.reshape(shape=(x.shape[0], h, w, p, p, hid_chans))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], hid_chans, h * p, w * p))

        x_LIDAR = x_LIDAR.reshape(shape=(x_LIDAR.shape[0], h, w, p_LIDAR, p_LIDAR, hid_chans_LIDAR))
        x_LIDAR = torch.einsum('nhwpqc->nchpwq', x_LIDAR)
        imgs_LIDAR = x_LIDAR.reshape(shape=(x_LIDAR.shape[0], hid_chans_LIDAR, h * p_LIDAR, w * p_LIDAR))
        return imgs, imgs_LIDAR

    def random_masking(self, x, x_LIDAR, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_visible = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))


        noise_LIDAR = torch.rand(N, L, device=x.device)
        ids_shuffle_LIDAR = torch.argsort(noise_LIDAR, dim=1)  # ascend: small is keep, large is remove
        ids_restore_LIDAR = torch.argsort(ids_shuffle_LIDAR, dim=1)

        # keep the first subset
        ids_keep_LIDAR = ids_restore_LIDAR[:, :len_keep]
        x_LIDAR_visible = torch.gather(x_LIDAR, dim=1, index=ids_keep_LIDAR.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        mask_LIDAR = torch.ones([N, L], device=x.device)
        mask_LIDAR[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)
        mask_LIDAR = torch.gather(mask_LIDAR, dim=1, index=ids_restore_LIDAR)
        return x_visible, x_LIDAR_visible, mask, ids_restore, mask_LIDAR, ids_restore_LIDAR

    def preprocessing(self, x, x_LIDAR, mask_ratio):
        # embed patches
        x = self.dimen_redu(x)
        x = self.patch_embed(x)

        x_LIDAR = self.dimen_redu_LIDAR(x_LIDAR)
        x_LIDAR = self.patch_embed_LIDAR(x_LIDAR)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]
        x_LIDAR = x_LIDAR + self.pos_embed_LIDAR[:, 1:, :]

        # masking: length -> length * mask_ratio
        x_visible, x_LIDAR_visible, mask, ids_restore, mask_LIDAR, ids_restore_LIDAR = self.random_masking(x, x_LIDAR, mask_ratio)
        return x_visible, x_LIDAR_visible, mask, ids_restore, mask_LIDAR, ids_restore_LIDAR

    def forward_encoder(self, x, x_LIDAR, t):
        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)

        t_token = self.t_embedder(torch.squeeze(t))
        t_token = torch.unsqueeze(t_token, dim=1)
        cls_tokens = cls_tokens + t_token

        x = torch.cat((cls_tokens, x), dim=1)

        cls_token_LIDAR = self.cls_token_LIDAR + self.pos_embed_LIDAR[:, :1, :]
        cls_tokens_LIDAR = cls_token_LIDAR.expand(x_LIDAR.shape[0], -1, -1)

        t_token_LIDAR = self.t_embedder_LIDAR(torch.squeeze(t))
        t_token_LIDAR = torch.unsqueeze(t_token_LIDAR, dim=1)
        cls_tokens_LIDAR = cls_tokens_LIDAR + t_token_LIDAR

        x_LIDAR = torch.cat((cls_tokens_LIDAR, x_LIDAR), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        for blk_LIDAR in self.blocks:
            x_LIDAR = blk_LIDAR(x_LIDAR)
        x_LIDAR = self.norm(x_LIDAR)

        return x, x_LIDAR

    def forward_decoder_A(self, x, ids_restore):
        # embed token
        x = self.decoder_embed(x)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # add pos embed
        x = x + self.decoder_pos_embed[:, :x.shape[1], :]

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        x = self.projection_de(x)
        return x

    def forward_decoder_B(self, x_LIDAR, ids_restore_LIDAR):
        # embed token
        x_LIDAR = self.decoder_embed_LIDAR(x_LIDAR)

        # append mask tokens to sequence
        mask_tokens_LIDAR = self.mask_token_LIDAR.repeat(x_LIDAR.shape[0], ids_restore_LIDAR.shape[1] + 1 - x_LIDAR.shape[1], 1)
        x_LIDAR_ = torch.cat([x_LIDAR[:, 1:, :], mask_tokens_LIDAR], dim=1)  # no cls token
        x_LIDAR_ = torch.gather(x_LIDAR_, dim=1, index=ids_restore_LIDAR.unsqueeze(-1).repeat(1, 1, x_LIDAR.shape[2]))  # unshuffle
        x_LIDAR = torch.cat([x_LIDAR[:, :1, :], x_LIDAR_], dim=1)  # append cls token

        # add pos embed
        x_LIDAR = x_LIDAR + self.decoder_pos_embed_LIDAR[:, :x_LIDAR.shape[1], :]

        # apply Transformer blocks
        for blk_LIDAR in self.decoder_blocks_LIDAR:
            x_LIDAR = blk_LIDAR(x_LIDAR)
        x_LIDAR = self.decoder_norm_LIDAR(x_LIDAR)

        x_LIDAR = self.projection_de_LIDAR(x_LIDAR)
        return x_LIDAR


    def reconstruction(self, x, x_LIDAR):
        x = self.decoder_pred(x)
        x_LIDAR = self.decoder_pred_LIDAR(x_LIDAR)
        # # remove cls token
        x = x[:, 1:, :]
        x_LIDAR = x_LIDAR[:, 1:, :]


        x, x_LIDAR = self.unpatchify(x, x_LIDAR)
        x = self.dimen_expa(x)
        x_LIDAR = self.dimen_expa_LIDAR(x_LIDAR)

        pred_Reconstruction, pred_LIDAR_Reconstruction = self.patchify(x, x_LIDAR)
        return x, x_LIDAR, pred_Reconstruction, pred_LIDAR_Reconstruction

    def forward_classification(self, x, x_LIDAR):
        if self.global_pool:
            feat = x[:, 1:, :].mean(dim=1)  # global pool without cls token
        else:
            feat = x[:, 0, :]  # with cls token

        if self.global_pool_LIDAR:
            feat_LIDAR = x_LIDAR[:, 1:, :].mean(dim=1)  # global pool without cls token
        else:
            feat_LIDAR = x_LIDAR[:, 0, :]  # with cls token
        feat_all = torch.cat((feat, feat_LIDAR),dim=1)
        logits = self.cls_head(feat_all)
        return logits
    
    def Reconstruction_loss(self, imgs, pred, imgs_LIDAR, pred_LIDAR, mask, mask_LIDAR):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove, 
        """
        target, target_LIDAR = self.patchify(imgs, imgs_LIDAR)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch
        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches

        if self.norm_pix_loss_LIDAR:
            mean_LIDAR = target_LIDAR.mean(dim=-1, keepdim=True)
            var_LIDAR = target_LIDAR.var(dim=-1, keepdim=True)
            target_LIDAR = (target_LIDAR - mean_LIDAR) / (var_LIDAR + 1.e-6)**.5

        loss_LIDAR = (pred_LIDAR - target_LIDAR) ** 2
        loss_LIDAR = loss_LIDAR.mean(dim=-1)  # [N, L], mean loss per patch
        loss_LIDAR = (loss_LIDAR * mask_LIDAR).sum() / mask_LIDAR.sum()  # mean loss on removed patches

        loss_all = loss + loss_LIDAR
        return loss_all

    def forward(self, imgs, imgs_LIDAR, t, y, mask_ratio):
        #preprocessing
        x_visible, x_LIDAR_visible, mask, ids_restore, mask_LIDAR, ids_restore_LIDAR = self.preprocessing(imgs, imgs_LIDAR, mask_ratio)

        #visible_process
        feature_visible, feature_visible_LIDAR = self.forward_encoder(x_visible, x_LIDAR_visible, t)
        pred = self.forward_decoder_A(feature_visible, ids_restore)
        pred_LIDAR = self.forward_decoder_B(feature_visible_LIDAR, ids_restore_LIDAR)

        # Reconstruction branch
        pred_imgs, pred_imgs_LIDAR, pred_Reconstruction, pred_LIDAR_Reconstruction = self.reconstruction(pred, pred_LIDAR)  # [N, L, p*p*3]

        # Cross Reconstruction branch
        Cross_pred = self.forward_decoder_A(feature_visible_LIDAR, ids_restore_LIDAR)
        Cross_pred_LIDAR = self.forward_decoder_B(feature_visible, ids_restore)
        Cross_pred_imgs, Cross_pred_imgs_LIDAR, Cross_pred_Reconstruction, Cross_pred_LIDAR_Reconstruction = self.reconstruction(Cross_pred, Cross_pred_LIDAR)

        # Classification branch
        logits = self.forward_classification(feature_visible, feature_visible_LIDAR)
        return pred_imgs, pred_imgs_LIDAR, logits, mask, mask_LIDAR, Cross_pred_imgs, Cross_pred_imgs_LIDAR

def patchify(imgs, imgs_LIDAR, size):
    """
    imgs: (N, 3, H, W)
    x: (N, L, patch_size**2 *3)
    """
    p = size
    p_LIDAR = size
    # assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

    h = imgs.shape[2] // p
    w = imgs.shape[3] // p

    x = imgs.reshape(shape=(imgs.shape[0], imgs.shape[1], h, p, w, p))
    x = torch.einsum('nchpwq->nhwpqc', x)
    x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * imgs.shape[1]))

    x_LIDAR = imgs_LIDAR.reshape(shape=(imgs_LIDAR.shape[0], imgs_LIDAR.shape[1], h, p_LIDAR, w, p_LIDAR))
    x_LIDAR = torch.einsum('nchpwq->nhwpqc', x_LIDAR)
    x_LIDAR = x_LIDAR.reshape(shape=(imgs_LIDAR.shape[0], h * w, p_LIDAR**2 * imgs_LIDAR.shape[1]))
    return x, x_LIDAR

def DDPM_LOSS(model_output, model_output_LIDAR, target, target_LIDAR, mask , mask_LIDAR, size):
    model_output, model_output_LIDAR = patchify(model_output, model_output_LIDAR, size)
    target, target_LIDAR = patchify(target, target_LIDAR, size)

    loss_mse = ((target - model_output) ** 2).mean(dim=-1)
    loss_mse_LIDAR = ((target_LIDAR - model_output_LIDAR) ** 2).mean(dim=-1)
    loss_mse_m = (loss_mse * mask).sum() / mask.sum()
    loss_mse_LIDAR_m = (loss_mse_LIDAR * mask_LIDAR).sum() / mask_LIDAR.sum()
    visible = torch.zeros_like(mask)
    visible_LIDAR = torch.zeros_like(mask_LIDAR)
    zeros_mask = torch.eq(mask, 0)
    ones_mask = torch.logical_not(zeros_mask)
    visible[zeros_mask] = 1
    visible[ones_mask] = 0
    zeros_mask = torch.eq(mask_LIDAR, 0)
    ones_mask = torch.logical_not(zeros_mask)
    visible_LIDAR[zeros_mask] = 1
    visible_LIDAR[ones_mask] = 0
    loss_mse_v = (loss_mse * visible).sum() / visible.sum()
    loss_mse_LIDAR_v = (loss_mse_LIDAR * visible_LIDAR).sum() / visible_LIDAR.sum()

    # loss_mse = ((target - model_output) ** 2).mean(dim=-1).mean()
    # loss_mse_LIDAR = ((target_LIDAR - model_output_LIDAR) ** 2).mean(dim=-1).mean()
    return loss_mse_m, loss_mse_LIDAR_m, loss_mse_v, loss_mse_LIDAR_v

class vit_HSI_LIDAR(nn.Module):
    """ Masked Autoencoder's'backbone
    """

    def __init__(self, img_size=(224, 224), patch_size=16, num_classes=1000, in_chans=3, in_chans_LIDAR = 1, hid_chans=32,
                 hid_chans_LIDAR=128,embed_dim=1024, depth=24, num_heads=16, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False, global_pool=False):
        super().__init__()
        self.patch_size = patch_size

        # --------------------------------------------------------------------------
        # HSI
        # MAE encoder specifics
        self.dimen_redu = nn.Sequential(
            nn.Conv2d(in_chans, hid_chans, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(hid_chans),
            nn.ReLU(),

            nn.Conv2d(hid_chans, hid_chans, 1, 1, 0, bias=True),
            nn.BatchNorm2d(hid_chans),
            nn.ReLU(),
        )

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_embed = PatchEmbed(img_size, patch_size, hid_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.t_embedder = TimestepEmbedder(embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim),
                                      requires_grad=True)  # fixed sin-cos embedding

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=drop_path_rate,
                  qkv_bias=True, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        self.head = nn.Linear(embed_dim * 2, num_classes, bias=True)

        self.global_pool = global_pool
        if self.global_pool:
            self.fc_norm = norm_layer(embed_dim)
            del self.norm

        # LIDAR
        # MAE encoder specifics
        self.dimen_redu_LIDAR = nn.Sequential(
            nn.Conv2d(in_chans_LIDAR, hid_chans_LIDAR, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(hid_chans_LIDAR),
            nn.ReLU(),

            nn.Conv2d(hid_chans_LIDAR, hid_chans_LIDAR, 1, 1, 0, bias=True),
            nn.BatchNorm2d(hid_chans_LIDAR),
            nn.ReLU(),
        )

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_embed_LIDAR = PatchEmbed(img_size, patch_size, hid_chans_LIDAR, embed_dim)
        num_patches = self.patch_embed_LIDAR.num_patches

        self.cls_token_LIDAR = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.t_embedder_LIDAR = TimestepEmbedder(embed_dim)
        self.pos_embed_LIDAR = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim),
                                      requires_grad=True)  # fixed sin-cos embedding

        self.blocks_LIDAR = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=drop_path_rate,
                  qkv_bias=True, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm_LIDAR = norm_layer(embed_dim)
        self.global_pool_LIDAR = global_pool
        if self.global_pool_LIDAR:
            self.fc_norm_LIDAR = norm_layer(embed_dim)
            del self.norm_LIDAR

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches ** .5),
                                            cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)

        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed_LIDAR = get_2d_sincos_pos_embed(self.pos_embed_LIDAR.shape[-1], int(self.patch_embed_LIDAR.num_patches ** .5),
                                            cls_token=True)
        self.pos_embed_LIDAR.data.copy_(torch.from_numpy(pos_embed_LIDAR).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w_LIDAR = self.patch_embed_LIDAR.proj.weight.data
        torch.nn.init.xavier_uniform_(w_LIDAR.view([w_LIDAR.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token_LIDAR, std=.02)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder_LIDAR.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder_LIDAR.mlp[2].weight, std=0.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x, x_LIDAR, t):
        x = self.dimen_redu(x)
        x_LIDAR = self.dimen_redu_LIDAR(x_LIDAR)

        # embed patches
        x = self.patch_embed(x)
        x_LIDAR = self.patch_embed_LIDAR(x_LIDAR)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]
        x_LIDAR = x_LIDAR + self.pos_embed_LIDAR[:, 1:, :]

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)

        t_token = self.t_embedder(t)
        t_token = torch.unsqueeze(t_token, dim=1)
        cls_tokens = cls_tokens + t_token

        x = torch.cat((cls_tokens, x), dim=1)


        cls_token_LIDAR = self.cls_token_LIDAR + self.pos_embed_LIDAR[:, :1, :]
        cls_tokens_LIDAR = cls_token_LIDAR.expand(x_LIDAR.shape[0], -1, -1)

        t_token_LIDAR = self.t_embedder_LIDAR(t)
        t_token_LIDAR = torch.unsqueeze(t_token_LIDAR, dim=1)
        cls_tokens_LIDAR = cls_tokens_LIDAR + t_token_LIDAR

        x_LIDAR = torch.cat((cls_tokens_LIDAR, x_LIDAR), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        if self.global_pool:
            x = x[:, 1:, :].mean(dim=1)  # global pool without cls token
            outcome = self.fc_norm(x)
        else:
            x = self.norm(x)
            outcome = x[:, 0]

        for blk_LIDAR in self.blocks:
            x_LIDAR = blk_LIDAR(x_LIDAR)
        if self.global_pool_LIDAR:
            x_LIDAR = x_LIDAR[:, 1:, :].mean(dim=1)  # global pool without cls token
            outcome_LIDAR = self.fc_norm(x_LIDAR)
        else:
            x_LIDAR = self.norm(x_LIDAR)
            outcome_LIDAR = x_LIDAR[:, 0]

        outcome_all = torch.cat((outcome, outcome_LIDAR),dim = 1)
        return outcome_all

    def forward(self, x, x_LIDAR, t):
        x = self.forward_features(x, x_LIDAR, t)
        x = self.head(x)
        return x


def mae_vit_HSIandLIDAR_patch3(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=1,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def vit_HSI_LIDAR_patch3(**kwargs):
    model = vit_HSI_LIDAR(
        patch_size=1,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model
