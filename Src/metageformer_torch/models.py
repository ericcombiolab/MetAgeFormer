import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Callable, Union, Dict
from .mask_utils import _generate_mask_matrix_VocabFree
from .checkpoint import load_distilled_checkpoint, save_distilled_checkpoint
from einops import repeat

import math


class MetAgeFormer_EmbeddingLayer(nn.Module):
    def __init__(self, Model_conf:dict, EmbeddingModule_conf:dict, b:int=100):
        
        super(MetAgeFormer_EmbeddingLayer, self).__init__()

        if "prior_embs" in EmbeddingModule_conf:
            raise ValueError(
                "prior_embs is no longer supported. "
                "Use EmbeddingModule_conf with n_vocabs['identifier'] only."
            )

        # metabolite symbol embeddings
        self.embs_ident_layer = nn.Embedding(
            num_embeddings=EmbeddingModule_conf["n_vocabs"]["identifier"],
            embedding_dim=Model_conf["d_model"],
        )

        # special tokens
        self.cls_emb = nn.Parameter(torch.randn(1, 1, Model_conf["d_model"]))       
        self.pad_emb = nn.Parameter(torch.randn(Model_conf["d_model"]))
        self.mask_emb = nn.Parameter(torch.randn(Model_conf["d_model"])) 


        # continous concentration values to value embeddings
        # ref to '8.Embedding module' of the supplementary file in scFoundation: https://www.nature.com/articles/s41592-024-02305-7
        self.lookup = nn.Linear(b, Model_conf["d_model"], bias=False)   # concentration values: loop-up table for value transformation              
        self.w1 = nn.Linear(1, b, bias=False)                           
        self.w2 = nn.Linear(b, b, bias=False)
        self.alpha = nn.Parameter(torch.randn(b)) 
        self.leakyRelu = nn.LeakyReLU()   
        self.softmax = nn.Softmax(dim=-1)

        
    def forward(self, inputs, add_cls:bool=True):
        
        ############ concentration value embedding ############
        x_conc = torch.unsqueeze(inputs['input_ids']['concentration'], dim=-1) 
        x_conc = torch.where(torch.isnan(x_conc), torch.tensor(0.0, dtype=x_conc.dtype), x_conc) # avoid error caused by NaN
    
        v1 = self.leakyRelu(self.w1(x_conc))     
        v2 = self.w2(v1) + self.alpha*v1
        v3 = self.softmax(v2)
        x_conc = self.lookup(v3)

        mask_indices = inputs['masking_mask'] == 1                                               # replace mask tokens with mask embedding
        x_conc = torch.where(mask_indices.unsqueeze(-1), self.mask_emb, x_conc)


        ############ metabolite symbol embedding ############
        x_metabo = self.embs_ident_layer(inputs['input_ids']['identifier'])


        ############ final embedding: model input ############
        merged_embs = torch.mean( torch.stack((x_metabo, x_conc), dim=-1) , dim=-1)               # merge concentration & metabolite embeddings  


        mask_indices = inputs['padding_mask'] == 1                                                # replace pad tokens with pad embedding
        merged_embs = torch.where(mask_indices.unsqueeze(-1), self.pad_emb, merged_embs)

    
        batch_size = inputs['input_ids']['concentration'].size(0)                                 # create a tensor for CLS embeddings: 
        if add_cls == True:                                                                       # ref to ViT: https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/vit.py
            cls_embs = repeat(self.cls_emb, '1 1 d -> b 1 d', b = batch_size)           
            merged_embs = torch.cat((cls_embs, merged_embs), dim=1)

        return merged_embs


class MetAgeFormer_Block(nn.Module):
    def __init__(self, d_ff:int=256, d_model:int=256, n_heads:int=4, dropout:float=0.3,
                 need_weights:bool=False, average_attn_weights:bool=False,
                 activation: Union[str, Callable[[Tensor], Tensor]] = nn.ReLU()):
        super(MetAgeFormer_Block, self).__init__()
        '''
        transformer block
        refer to https://pytorch.org/docs/stable/_modules/torch/nn/modules/transformer.html#TransformerDecoderLayer
        '''

        # model configuration
        self.d_ff = d_ff
        self.d_model= d_model
        self.n_heads = n_heads
        self.dropout_rate = dropout
        self.need_weights = need_weights
        self.average_attn_weights = average_attn_weights
    
        # layers
        self.MaskedMHA = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, batch_first=True, bias=False)
        self.LayerNorm = nn.LayerNorm(normalized_shape=d_model, eps=1e-6)
        self.fc_ff_1 = nn.Linear(d_model, d_ff)
        self.activation = nn.ReLU()
        self.fc_ff_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(p=dropout)

        # Legacy string support for activation function.
        if isinstance(activation, str):
            self.activation = _get_activation_fn(activation)
        else:
            self.activation = activation
  


    def forward(self, x:Tensor, attn_mask: Optional[Tensor]=None, padding_mask: Optional[Tensor]=None) -> Tensor:
        '''
        attn_mask (Optional[Tensor]): 
            If specified, a 2D or 3D mask preventing attention to certain positions.
            A 2D mask will be broadcasted across the batch while a 3D mask allows for a different mask for each entry in the batch. 
            Binary and float masks are supported. For a binary mask, a True value indicates that the corresponding position is not allowed to attend. 
            For a float mask, the mask values will be added to the attention weight. If both attn_mask and key_padding_mask are supplied, their types should match.
        '''

        x_residue = x
        x, attn = self._sa_block(x, attn_mask, padding_mask)
        x = self.LayerNorm(x_residue + x)
        x = self.LayerNorm(x + self._ff_block(x))
        return x, attn


    # self-attention block
    def _sa_block(self, x:Tensor, attn_mask:Optional[Tensor], key_padding_mask:Optional[Tensor], is_causal:bool=False) -> Tensor:
        x, attn = self.MaskedMHA(x, x, x,
                           attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask,
                           is_causal=is_causal,
                           need_weights=self.need_weights,
                           average_attn_weights=self.average_attn_weights)
        return self.dropout(x), attn



    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.fc_ff_2(self.dropout(self.activation(self.fc_ff_1(x))))
        return self.dropout(x)


def _get_activation_fn(activation: str) -> Callable[[Tensor], Tensor]:
    if activation == "relu":
        return nn.ReLU()
    elif activation == "gelu":
        return nn.GELU()

    raise RuntimeError(f"activation should be relu/gelu, not {activation}")



class PredictionHeadTransform(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.dense = nn.Linear(d_model, d_model)
        self.transform_act_fn = nn.ReLU()
        self.LayerNorm = nn.LayerNorm(d_model)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states
    

class ConcentrationPredictionHead(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        
        self.transform = PredictionHeadTransform(d_model)
        self.decoder = nn.Linear(d_model, 1)

        # # concentration values should be positive
        # self.softplus = nn.Softplus()

    def _tie_weights(self):
        self.decoder.bias = self.bias

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)
        logit = torch.squeeze(hidden_states, dim=-1)
        return logit




###### Models


    


        
###### Models
class MetAgeFormer_Model(nn.Module):
    def __init__(self,EmbeddingModule_conf, Model_conf, max_input_tokens:int=512):
        super(MetAgeFormer_Model, self).__init__()
        '''
        Basic architecture of MetAgeFormer model
        '''      
        self.d_model = Model_conf["d_model"]
        self.n_blocks = Model_conf["n_blocks"]
        self.attn_mode = Model_conf["attn_mode"]
        self.max_input_tokens = max_input_tokens

        self.emb_layer = MetAgeFormer_EmbeddingLayer(Model_conf,
                                                 EmbeddingModule_conf,
                                                 b=100                                             # b=100: pre-defined in scFoundation
                                                )

        self.blocks = nn.ModuleList([MetAgeFormer_Block(d_ff=Model_conf["d_ff"],
                                                        d_model=Model_conf["d_model"],
                                                        n_heads=Model_conf["n_heads"],
                                                        dropout=Model_conf["dropout"],
                                                        activation=Model_conf["activation"],
                                                        need_weights=Model_conf["need_weights"],
                                                        average_attn_weights=Model_conf["average_attn_weights"])
                                            for _ in range(Model_conf["n_blocks"])])
        

    def forward(self, inputs):
        # limited the number of input tokens
        if inputs['padding_mask'].shape[1] > self.max_input_tokens:
            raise ValueError(f"the maximum number of input tokens is {self.max_input_tokens} while got {inputs['padding_mask'].shape[1]} tokens")
        
        # mask
        padding_mask = inputs['padding_mask']
        padding_mask = torch.cat((torch.zeros(padding_mask.size(0), 1).to(padding_mask.device), padding_mask), dim=1) # add zeros at the first column to represent <CLS>; 0:unmasked
        padding_mask = padding_mask.bool()                                                               # convert 0/1 to boolean type to satisfy the requirement of nn.MultiheadAttention()
        
        if self.attn_mode == 'mixdirect_mask':                                                      # 1:masked, 0:unmasked
            attn_mask = inputs['mixdirect_mask'].bool()                     
        elif self.attn_mode == 'bidirect_mask':
            attn_mask = torch.zeros(inputs['mixdirect_mask'].shape).bool().to(padding_mask.device)  # bi-directional attention
        else:
            raise TypeError(f"attn_mode not support {self.attn_mode}")

        # embedding
        x = self.emb_layer(inputs)

        # transformer block forward
        attn_each_block = []
        for i in range(self.n_blocks):
            x, attn = self.blocks[i](x, attn_mask=attn_mask, padding_mask=padding_mask)             # 'is_causal' set to False to allow MHA to accept 'attn_mask'
            attn_each_block.append(attn)

        return x, (attn_each_block,None)



class MetAgeFormer_ForPreTrain(nn.Module):        

    def __init__(self, EmbeddingModule_conf, Model_conf):
        super(MetAgeFormer_ForPreTrain, self).__init__()
        '''
        MetAgeFormer Language Model for pre-training
        '''
        self.n_heads=Model_conf['n_heads']
        self.d_model=Model_conf['d_model']
        self.metageformer_model = MetAgeFormer_Model(EmbeddingModule_conf, Model_conf)
        self.conc_predictor =  ConcentrationPredictionHead(Model_conf['d_model'])

        
    def forward(self, inputs):
        ouputs = {}                                                     # output object

        inputs = self.generate_mixdirect_mask(inputs)                   # mixdirect mask
        h, _ = self.metageformer_model(inputs)

        h_sample = h[:,0,:]                                             # sample embedding
        ouputs['embs'] = h_sample

        logit = self.conc_predictor(h)                                  # concentration prediction  
        ouputs['logit_conc'] = logit

        return ouputs
    


    def generate_mixdirect_mask(self, inputs) -> Tensor:
        # masked tokens can only 'see' the known tokens and itself in the self-attention matrix
        batch_size = inputs['input_ids']['concentration'].size(0)
        mask_matrix =inputs['masking_mask']
        mask_matrix = torch.cat((torch.zeros(batch_size, 1).to(mask_matrix.device), mask_matrix), dim=1) # add zeros at the first column to represent <CLS>; 0:unmasked
        mask_matrix = _generate_mask_matrix_VocabFree(inputs['input_ids']['concentration'],
                                                        mask_matrix=mask_matrix)

        # expand mask matrix for multi-head
        # ref to: https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html -> attn_mask
        N = inputs['padding_mask'].shape[0]
        # seq_len = inputs['padding_mask'].shape[1]
        seq_len = mask_matrix.shape[1]
        mask_matrix = mask_matrix.unsqueeze(1).expand(-1, self.n_heads, -1, -1)
        mask_matrix = mask_matrix.reshape(N * self.n_heads, seq_len, seq_len) 

        inputs['mixdirect_mask'] = mask_matrix
        return inputs


    def save_pretrained(self, save_path):
        # Keep MULTITASK_HEADS key for schema compatibility with existing checkpoints.
        torch.save({'METAGEFORMER':self.metageformer_model.state_dict(),
                    'CONCENTRATION_PREDICTOR':self.conc_predictor.state_dict(), 
                    'MULTITASK_HEADS':{}}, 
                    save_path) 
               

    def from_pretrained(self, model_path):
        model_weights = torch.load(  model_path )
        self.metageformer_model.load_state_dict(model_weights['METAGEFORMER'])
        self.conc_predictor.load_state_dict(model_weights['CONCENTRATION_PREDICTOR'])




class MetAgeFormer_Pretrained(nn.Module):        

    def __init__(self, EmbeddingModule_conf, Model_conf, Model_path):
        super(MetAgeFormer_Pretrained, self).__init__()
        '''
        MetAgeFormer Foundation Model
        '''
        self.n_heads=Model_conf['n_heads']
        self.d_model=Model_conf['d_model']

        self.metageformer_model = MetAgeFormer_Model(EmbeddingModule_conf, Model_conf)
        self.conc_predictor =  ConcentrationPredictionHead(Model_conf['d_model'])

        if 'multitask_config' in Model_conf:
            raise ValueError(
                "This checkpoint/config enables multitask pretrain heads, which are no longer supported. "
                "Use an MLM pretrained config without multitask_config."
            )

        # load pre-trained weights
        self.from_pretrained(Model_path)


        
    def forward(self, inputs):
        ouputs = {}                                                     # output object

        inputs = self.generate_mixdirect_mask(inputs)                   # mixdirect mask

        h, attn_ = self.metageformer_model(inputs)
        attn = attn_[0]                                                 # attention from encoder part
        
        h_sample = h[:,0,:]                                             # sample embedding
        ouputs['embs'] = h_sample

        logit = self.conc_predictor(h)                                  # concentration prediction  
        ouputs['logit_conc'] = logit
        ouputs['attn'] = attn
        
        ouputs['metabolite embs'] = h[:,1:,:] 

        return ouputs
    

    def generate_mixdirect_mask(self, inputs) -> Tensor:
        # masked tokens can only 'see' the known tokens and itself in the self-attention matrix
        batch_size = inputs['input_ids']['concentration'].size(0)
        mask_matrix =inputs['masking_mask']
        mask_matrix = torch.cat((torch.zeros(batch_size, 1).to(mask_matrix.device), mask_matrix), dim=1) # add zeros at the first column to represent <CLS>; 0:unmasked
        mask_matrix = _generate_mask_matrix_VocabFree(inputs['input_ids']['concentration'],
                                                        mask_matrix=mask_matrix)

        # expand mask matrix for multi-head
        # ref to: https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html -> attn_mask
        N = inputs['padding_mask'].shape[0]
        # seq_len = inputs['padding_mask'].shape[1]
        seq_len = mask_matrix.shape[1]
        mask_matrix = mask_matrix.unsqueeze(1).expand(-1, self.n_heads, -1, -1)
        mask_matrix = mask_matrix.reshape(N * self.n_heads, seq_len, seq_len) 

        inputs['mixdirect_mask'] = mask_matrix
        return inputs



    def from_pretrained(self, model_path):
        model_weights = torch.load(  model_path )
        self.metageformer_model.load_state_dict(model_weights['METAGEFORMER'])
        self.conc_predictor.load_state_dict(model_weights['CONCENTRATION_PREDICTOR'])

  


############################################################
############### Lightweight Models (blood-token Transformer)
############################################################

class MetAgeFormer_Lightweight(nn.Module):
    """Lightweight student: per-biomarker token Transformer on blood panel vectors.

    Each scalar biomarker -> Linear(1, d_model) token + fixed positional embedding.
    Missing values (NaN) -> learnable mask embedding with key_padding_mask (no zero-fill).
    CLS token at index 0 is the sample embedding fed to downstream heads.
    """

    def __init__(self, model_conf: Dict):
        super().__init__()
        self.n_features = int(model_conf["n_features"])
        self.d_model = int(model_conf["d_model"])
        n_heads = int(model_conf.get("n_heads", 4))
        n_blocks = int(model_conf.get("n_blocks", 2))
        d_ff = int(model_conf.get("d_ff", max(256, self.d_model * 2)))
        dropout = float(model_conf.get("dropout", 0.1))
        if self.d_model % n_heads != 0:
            raise ValueError(f"d_model={self.d_model} must be divisible by n_heads={n_heads}")

        self.value_proj = nn.Linear(1, self.d_model)
        self.pos_emb = nn.Embedding(self.n_features, self.d_model)
        self.cls_emb = nn.Parameter(torch.randn(1, 1, self.d_model) * 0.02)
        self.mask_emb = nn.Parameter(torch.randn(1, 1, self.d_model) * 0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_blocks)
        self.out_norm = nn.LayerNorm(self.d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, n_features), may contain NaN for missing biomarkers."""
        if x.ndim != 2 or x.shape[1] != self.n_features:
            raise ValueError(f"Expected x shape (B, {self.n_features}), got {tuple(x.shape)}")

        missing = torch.isnan(x)
        x_safe = torch.where(missing, torch.zeros_like(x), x)
        tokens = self.value_proj(x_safe.unsqueeze(-1))
        pos = self.pos_emb.weight.unsqueeze(0).expand(x.shape[0], -1, -1)
        tokens = tokens + pos
        tokens = torch.where(missing.unsqueeze(-1), self.mask_emb.expand_as(tokens), tokens)

        cls = self.cls_emb.expand(x.shape[0], 1, -1)
        h = torch.cat([cls, tokens], dim=1)

        key_padding_mask = torch.cat(
            [
                torch.zeros(x.shape[0], 1, dtype=torch.bool, device=x.device),
                missing,
            ],
            dim=1,
        )
        h = self.encoder(h, src_key_padding_mask=key_padding_mask)
        return self.out_norm(h[:, 0, :])


def inverse_softplus(value: float) -> float:
    return math.log(math.expm1(value))


def deep_gompertz_nll_loss(
    alpha_i: torch.Tensor,
    gamma_i: torch.Tensor,
    log_age_effect: torch.Tensor,
    time: torch.Tensor,
    event: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    alpha_i = alpha_i.reshape(-1).clamp(min=-30.0, max=30.0)
    gamma_i = gamma_i.reshape(-1).clamp_min(eps)
    log_age_effect = log_age_effect.reshape(-1).clamp(min=-30.0, max=30.0)
    time = time.reshape(-1).clamp_min(eps)
    event = event.reshape(-1)

    gamma_time = (gamma_i * time).clamp(max=50.0)
    log_time_integral = torch.log(torch.expm1(gamma_time).clamp_min(eps)) - torch.log(gamma_i)
    log_cum_hazard = log_age_effect + alpha_i + log_time_integral
    cum_hazard = torch.exp(log_cum_hazard.clamp(min=-30.0, max=30.0))
    log_hazard = log_age_effect + alpha_i + gamma_i * time

    return -(event * log_hazard - cum_hazard).mean()


class DeepGompertzMetabolomicAgeHead(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int = 64,
        dropout: float = 0.1,
        t_window: float = 10.0,
        age_scale: float = 1.0,
        baseline_params: Optional[Dict[str, float]] = None,
        init_alpha: float = -10.5,
        init_gamma: float = 0.09,
        init_beta_age: float = 0.1,
        gamma_min: float = 1e-6,
    ):
        super().__init__()
        self.embedding_dim = int(embedding_dim)
        self.hidden_dim = int(hidden_dim)
        self.dropout = float(dropout)
        self.gamma_min = float(gamma_min)

        self.alpha_head = nn.Sequential(
            nn.Linear(self.embedding_dim, self.hidden_dim),
            nn.Tanh(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, 1),
        )
        self.gamma_head = nn.Sequential(
            nn.Linear(self.embedding_dim, self.hidden_dim),
            nn.Tanh(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, 1),
        )
        self.beta_age = nn.Parameter(torch.tensor(float(init_beta_age)))

        self._initialize_heads(init_alpha=init_alpha, init_gamma=init_gamma)

        baseline_params = baseline_params or {
            "alpha_age_scale": -10.5,
            "gamma_age_scale": 0.09,
        }
        alpha_age_scale = baseline_params.get("alpha_age_scale", baseline_params.get("alpha_B", -10.5))
        gamma_age_scale = baseline_params.get("gamma_age_scale", baseline_params.get("gamma_B", 0.09))
        self.register_buffer("alpha_age_scale", torch.tensor(float(alpha_age_scale)))
        self.register_buffer("gamma_age_scale", torch.tensor(float(gamma_age_scale)))
        self.register_buffer("T", torch.tensor(float(t_window)))

    def _initialize_heads(self, init_alpha: float, init_gamma: float):
        for head in (self.alpha_head, self.gamma_head):
            for module in head:
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    nn.init.zeros_(module.bias)
        nn.init.zeros_(self.alpha_head[-1].weight)
        nn.init.constant_(self.alpha_head[-1].bias, float(init_alpha))
        nn.init.zeros_(self.gamma_head[-1].weight)
        nn.init.constant_(self.gamma_head[-1].bias, inverse_softplus(float(init_gamma)))

    def individual_parameters(self, embedding):
        alpha_i = self.alpha_head(embedding)
        gamma_i = F.softplus(self.gamma_head(embedding)) + self.gamma_min
        return alpha_i, gamma_i

    def log_age_effect(self, age):
        return self.beta_age * age.reshape(-1, 1)

    def mortality_risk(self, alpha_i, gamma_i, log_age_effect):
        gamma_i = gamma_i.clamp_min(self.gamma_min)
        gamma_time = (gamma_i * self.T).clamp(max=50.0)
        log_time_integral = torch.log(torch.expm1(gamma_time).clamp_min(1e-8)) - torch.log(gamma_i)
        log_hazard_window = log_age_effect + alpha_i + log_time_integral
        cum_hazard = torch.exp(log_hazard_window.clamp(min=-30.0, max=30.0))
        return 1.0 - torch.exp(-cum_hazard)

    def metabolomic_age(self, mortality_risk):
        risk = mortality_risk.clamp(min=1e-7, max=1.0 - 1e-7)
        denom = torch.exp(self.alpha_age_scale) * torch.expm1((self.gamma_age_scale * self.T).clamp(max=50.0))
        numerator = -self.gamma_age_scale * torch.log1p(-risk)
        age_years = torch.log(numerator / denom.clamp_min(1e-8)) / self.gamma_age_scale.clamp_min(1e-8)
        return age_years

    def forward(self, embedding, age):
        alpha_i, gamma_i = self.individual_parameters(embedding)
        log_age_effect = self.log_age_effect(age)
        linear_predictor = log_age_effect + alpha_i
        mortality_risk = self.mortality_risk(alpha_i, gamma_i, log_age_effect)
        metabolomic_age = self.metabolomic_age(mortality_risk)
        chronological_age = age.reshape(-1, 1)
        return {
            "linear_predictor": linear_predictor,
            "log_age_effect": log_age_effect,
            "alpha_i": alpha_i,
            "gamma_i": gamma_i,
            "mortality_risk_10y": mortality_risk,
            "metabolomic_age": metabolomic_age,
            "age_gap": metabolomic_age - chronological_age,
        }

    def config_dict(self) -> Dict[str, float]:
        return {
            "model_type": "DeepGompertz",
            "embedding_dim": self.embedding_dim,
            "hidden_dim": self.hidden_dim,
            "dropout": self.dropout,
            "t_window": float(self.T.detach().cpu()),
            "gamma_min": self.gamma_min,
            "alpha_age_scale": float(self.alpha_age_scale.detach().cpu()),
            "gamma_age_scale": float(self.gamma_age_scale.detach().cpu()),
        }


class DeepGompertzEndToEndModel(nn.Module):
    def __init__(
        self,
        embedding_module_conf: Dict,
        model_conf: Dict,
        baseline_params: Dict[str, float],
        hidden_dim: int = 64,
        dropout: float = 0.1,
        t_window: float = 10.0,
        init_alpha: float = -10.5,
        init_gamma: float = 0.09,
        init_beta_age: float = 0.1,
        gamma_min: float = 1e-6,
    ):
        super().__init__()
        self.n_heads = model_conf["n_heads"]
        self.d_model = model_conf["d_model"]
        self.backbone_config = dict(model_conf)
        self.metageformer_model = MetAgeFormer_Model(embedding_module_conf, model_conf)
        self.head = DeepGompertzMetabolomicAgeHead(
            embedding_dim=self.d_model,
            hidden_dim=hidden_dim,
            dropout=dropout,
            t_window=t_window,
            baseline_params=baseline_params,
            init_alpha=init_alpha,
            init_gamma=init_gamma,
            init_beta_age=init_beta_age,
            gamma_min=gamma_min,
        )

    def generate_mixdirect_mask(self, inputs) -> dict:
        batch_size = inputs["input_ids"]["concentration"].size(0)
        mask_matrix = inputs["masking_mask"]
        mask_matrix = torch.cat(
            (torch.zeros(batch_size, 1).to(mask_matrix.device), mask_matrix), dim=1
        )
        mask_matrix = _generate_mask_matrix_VocabFree(
            inputs["input_ids"]["concentration"],
            mask_matrix=mask_matrix,
        )
        n_samples = inputs["padding_mask"].shape[0]
        seq_len = mask_matrix.shape[1]
        mask_matrix = mask_matrix.unsqueeze(1).expand(-1, self.n_heads, -1, -1)
        mask_matrix = mask_matrix.reshape(n_samples * self.n_heads, seq_len, seq_len)
        inputs["mixdirect_mask"] = mask_matrix
        return inputs

    def forward(self, inputs, age):
        inputs = self.generate_mixdirect_mask(inputs)
        hidden, _ = self.metageformer_model(inputs)
        embedding = hidden[:, 0, :]
        return self.head(embedding, age)

    def config_dict(self) -> Dict:
        config = self.head.config_dict()
        config.update(
            {
                "model_type": "DeepGompertzEndToEnd",
                "training_mode": "from_scratch",
                "d_model": self.d_model,
                "n_heads": self.n_heads,
                "n_blocks": self.backbone_config.get("n_blocks"),
                "d_ff": self.backbone_config.get("d_ff"),
                "dropout_backbone": self.backbone_config.get("dropout"),
                "attn_mode": self.backbone_config.get("attn_mode"),
            }
        )
        return config


class DeepGompertzFullyFinetuneModel(DeepGompertzEndToEndModel):
    def __init__(self, pretrained_dir: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pretrained_dir = pretrained_dir

    def config_dict(self) -> Dict:
        config = super().config_dict()
        config["model_type"] = "DeepGompertzFullyFinetune"
        config["training_mode"] = "fully_finetune"
        config["pretrained_dir"] = self.pretrained_dir
        return config


class MetAgeFormer_Lightweight_DeepGompertz(nn.Module):
    """Lightweight blood-token Transformer + DeepGompertz head."""

    def __init__(
        self,
        model_conf: Dict,
        gompertz_head_config: Optional[Dict] = None,
        baseline_params: Optional[Dict[str, float]] = None,
    ):
        super().__init__()
        self.model_conf = dict(model_conf)
        self.d_model = int(model_conf["d_model"])
        self.lightweight_model = MetAgeFormer_Lightweight(model_conf)
        gompertz_head_config = gompertz_head_config or {}
        self.gompertz_head = DeepGompertzMetabolomicAgeHead(
            embedding_dim=self.d_model,
            hidden_dim=gompertz_head_config.get("hidden_dim", 64),
            dropout=gompertz_head_config.get("dropout", 0.1),
            t_window=gompertz_head_config.get("t_window", 10.0),
            baseline_params=baseline_params,
            init_alpha=gompertz_head_config.get("init_alpha", -10.5),
            init_gamma=gompertz_head_config.get("init_gamma", 0.09),
            init_beta_age=gompertz_head_config.get("init_beta_age", 0.1),
            gamma_min=gompertz_head_config.get("gamma_min", 1e-6),
        )

    def forward(self, x: torch.Tensor, age_years: torch.Tensor) -> Dict[str, torch.Tensor]:
        outputs: Dict[str, torch.Tensor] = {}
        embs = self.lightweight_model(x)
        outputs["embs"] = embs
        outputs.update(self.gompertz_head(embs, age_years))
        return outputs

    def _load_gompertz_head_weights(self, checkpoint_path: str, device: str = "cpu"):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        state_dict = checkpoint["state_dict"]
        self.gompertz_head.load_state_dict(state_dict, strict=False)

    def freeze_gompertz_head(self):
        for param in self.gompertz_head.parameters():
            param.requires_grad = False

    def unfreeze_gompertz_head(self):
        for param in self.gompertz_head.parameters():
            param.requires_grad = True

    def save_distilled(self, save_path: str):
        save_distilled_checkpoint(self.state_dict(), save_path)

    def from_distilled(self, save_path: str):
        payload = load_distilled_checkpoint(save_path)
        self.load_state_dict(payload["METAGEFORMER_DISTILLED"])
