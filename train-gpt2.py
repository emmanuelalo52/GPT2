from dataclasses import dataclass
from torch.nn import functional as F
import torch.nn as nn
import torch
import math
import inspect



@dataclass
class GPTConfig:
    block_size: int = 1024
    n_emb: int = 768
    vocab_size: int = 50257
    n_head: int = 12
    n_layers: int = 12


class SelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_emb % config.n_head == 0
        self.c_attn = nn.Linear(config.n_emb, 3 * config.n_emb)
        #output layer
        self.c_proj = nn.Linear(config.n_emb,config.n_emb)
        self.c_proj.GPT2_SCALE_INIT = 1
        #regularization
        self.n_head = config.n_head
        self.n_emb = config.n_emb
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size,config.block_size)).view(1,1,config.block_size,config.block_size))
    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        # nh is "number of heads", hs is "head size", and C (number of channels) = nh * hs
        # e.g. in GPT-2 (124M), n_head=12, hs=64, so nh*hs=C=768 channels in the Transformer
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_emb, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True) # flash attention
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        # output projection
        y = self.c_proj(y)
        return y


# class TanhGelu(nn.Module):
#     def forward(self,input):
#         return (0.5 * input) * (1.0 + torch.tanh(math.sqrt(2.0/math.pi)*(input + 0.044715 * torch.pow(input,3))))


class MLP(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.cfc = nn.Linear(config.n_emb, 4* config.n_emb)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4 * config.n_emb, config.n_emb)
        self.c_proj.GPT2_SCALE_INIT = 1

    def forward(self,x):
        x = self.cfc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class Block(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_emb)
        self.attn = SelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_emb)
        self.mlp = MLP(config)
    def forward(self,x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

"""NOTE: wte: token embeddings
        wpe: Positional Encodings"""

class GPT(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(wte = nn.Embedding(config.vocab_size,config.n_emb), #token embedding
                                         wpe = nn.Embedding(config.block_size,config.n_emb), #positon emedding
                                         hidden_layer = nn.ModuleList([Block(config) for _ in range(config.n_layers)]),
                                         ln_final = nn.LayerNorm(config.n_emb),)
        )
        self.n_heads = nn.Linear(config.n_emb,config.vocab_size,bias=False)
        #share weight of output embedding at the beginning of the layer and at the pre-softmax stage
        self.transformer.wte.weight = self.n_heads.weight
        #initalise parameters
        self.apply(self.__init__weights)

    def __init__weights(self,module):
        if isinstance(module,nn.Linear):
            std = 0.02
            if hasattr(module,'GPT_SCALE_INIT'):
                std *= (2 * self.config.n_layers) ** -0.5
            torch.nn.init.normal_(module.weight,mean=0.0,std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module,nn.Embedding):
            torch.nn.init.normal_(module.weight,mean=0.0,std=0.02)

    def forward(self, idx, targets=None):
        # idx is of shape (B, T)
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        # forward the token and posisition embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device) # shape (T)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (T, n_embd)
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (B, T, n_embd)
        x = tok_emb + pos_emb
        # forward the blocks of the transformer
        for block in self.transformer.hidden_layer:
            x = block(x)
        # forward the final layernorm and the classifier
        x = self.transformer.ln_final(x)
        logits = self.n_heads(x) # (B, T, vocab_size)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss
    @classmethod
    def from_pretrained(cls,model_type):
        assert model_type in {'gpt2','gpt2-medium','gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print("Loading weights from huggingface server: %s" % model_type)
        #list out all the hyperparemeters for respective models
        config_params = {
            'gpt2': dict(n_layer = 12, n_head = 12, n_embed = 768), #124M params
            'gpt2-medium': dict(n_layer = 24, n_head = 16, n_embed = 1024), #350M params
            'gpt2-large': dict(n_layer = 36, n_head = 20, n_embed = 1280), #774M params
            'gpt2-xl': dict(n_layers = 48, n_head = 25, n_embed = 1600), #1558 params
        }[model_type]
        config_params['vocab_size'] = 50257 # 50000 merges and 1 special token that delimits 
        config_params['block_size'] = 1024 
        #create our initializer
        config = GPTConfig(**config_params)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = model.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] #removes the register buffer from huggingface gpt model and uses our own masking
        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model
    def configure_optimizers(self, weight_decay, learning_rate, device_type):
        # start with all of the candidate parameters (that require grad)
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        print(f"using fused AdamW: {use_fused}")
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
        return optimizer
    
import tiktoken
class Dataloader:
    def __init__(self,B,T):
        self.B = B
        self.T = T
        with open('input.txt', 'r') as f:
            text = f.read()
        enc = tiktoken.get_encoding("gpt2")
        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens)
        print(f"loaded {len(self.tokens)} tokens")
        print(f"1 epoch = {len(self.tokens)//(B*T)} batches")
        #inital position
        self.current_pos = 0
    def next_batch(self):
        B,T = self.B, self.T
        buf = self.tokens[self.current_pos:self.current_pos+B*T+1]
        x = (buf[:-1]).view(B,T)
        y = (buf[1:]).view(B,T)
        self.current_pos += B * T
        if self.current_pos + (B*T+1) > len(self.tokens):
            self.current_pos = 0
        return x,y

#------------------------------------------------------------------------------------------------------------
#Generate(test) output
import time
#check if user has a GPU or Apple silicon
device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
elif hasattr(torch.backends,"mps") and torch.backends.mps.is_available():
    device = "mps"
print(f"Device: {device}")


torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.manual_seed(1337)

train_loader = Dataloader(B=4, T=32)

torch.set_float32_matmul_precision('high')

num_return_sequences = 5
max_length = 30

max_lr = 6e-4
max_steps = 50
warmup_step = 10
min_lr = max_lr * 0.1

def get_lr(step_size):
    if step_size < warmup_step:
        return max_lr * (step_size + 1 ) /  warmup_step
    if step_size > max_steps:
        return min_lr
    #cosine decay according to the gpt3 paper
    decay_ratio = (step_size-warmup_step) / (max_steps-warmup_step)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)
# model = GPT.from_pretrained('gpt2')
model = GPT(GPTConfig(vocab_size=50304))
model.to(device)
# model = torch.compile(model) # if you are using A100 and above
# logits,loss = model(x,y)

# print(loss)


#optimize our losses
# optimizer = torch.optim.AdamW(model.parameters(),lr=3e-4, betas=(0.9,0.95),eps=1e-8)
optimizer = model.configure_optimizers(weight_decay=0.1,learning_rate=6e-4,device_type=device)



for step in range(max_steps):
    t0 = time.time()
    x,y = train_loader.next_batch()
    x,y = x.to(device), y.to(device)
    optimizer.zero_grad()
    with torch.autocast(device_type=device,dtype=torch.float16):
        logits,loss = model(x,y)
        # import code; code.interact(local=locals())
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm(model.parameters(),1.0)
    # determine the step size 
    lr = get_lr(step)
    for params_group in optimizer.param_groups:
        params_group['lr'] = lr
    optimizer.step()
    torch.cuda.synchronize() #waits for CPU and GPU to sync before calculating processing time
    t1 = time.time()
    dt = (t1-t0)*1000
    tok_per_sec = (train_loader.B*train_loader.T)/(t1-t0)
    print(f"step {step:4d} | loss: {loss.item():.6f} | lr: {lr:.4e}| norm: {norm:.4f} | dt: {dt:.2f}ms | token/sec: {tok_per_sec:.2f}")
import sys; sys.exit(0)

model.eval() # test model without training


#prefix tokens

# enc = tiktoken.get_encoding('gpt2')
# text = "Hello, I am a generative model,"
# token  = enc.encode(text)
# token = torch.tensor(token, dtype=torch.long)
# token = token.unsqueeze(0).repeat(num_return_sequences, 1)
# x = token.to(device)


torch.manual_seed(42)
torch.cuda.manual_seed(42)
while x.size(1) < max_length:
    while torch.no_grad():
        logits = model(x)
        logits = logits[:,-1,:]
        probs = F.softmax(logits,dim=-1)
        topk_probs, topk_indices = torch.topk(probs, 50, dim=-1) #keep the top 50 probabilities
        ix = torch.multinomial(topk_probs,1)
        #map corresponding indices
        xcol = torch.gather(topk_indices,-1,ix)
        x = torch.cat((x,xcol),dim=-1)

#print all return sequences
for i in range(num_return_sequences):
    tokens = x[i,:max_length].to_list()
    decoded = enc.decode(tokens)
    print(">", decoded)