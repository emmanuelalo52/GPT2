from dataclasses import dataclass
from torch.nn import functional as F
import torch.nn as nn
import torch
import math
import inspect
from hellaswag import render_example, iterate_examples

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
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


import tiktoken
import numpy as np

def load_tokens(filename):
    npt = np.load(filename)
    ptt = torch.tensor(npt,dtype=torch.long)
    return ptt


class Dataloader:
    def __init__(self,B,T,process_rank, num_processes, split):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
        assert split in {'train','val'}
        # get the shard filenames
        data_root = "edu_fineweb10B"
        shards = os.listdir(data_root)
        shards = [s for s in shards if split in s]
        shards = sorted(shards)
        shards = [os.path.join(data_root, s) for s in shards]
        self.shards = shards
        assert len(shards) > 0, f"no shards found for split {split}"
        if master_process:
            print(f"found {len(shards)} shards for split {split}")
        self.reset()

    def reset(self):
        # state, init at shard zero
        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = self.B * self.T * self.process_rank



# For shakespare dataset--------------------------------------------------------------
    #     with open('input.txt', 'r') as f:
    #         text = f.read()
    #     enc = tiktoken.get_encoding("gpt2")
    #     tokens = enc.encode(text)
    #     self.tokens = torch.tensor(tokens)
    #     print(f"loaded {len(self.tokens)} tokens")
    #     print(f"1 epoch = {len(self.tokens)//(B*T)} batches")
    #     #inital position
    #     self.current_pos = self.B * self.T * self.process_rank
    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position+B*T+1]
        x = (buf[:-1]).view(B, T) # inputs
        y = (buf[1:]).view(B, T) # targets
        # advance the position in the tensor
        self.current_position += B * T * self.num_processes
        # if loading the next batch would be out of bounds, advance to next shard
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.current_position = B * T * self.process_rank
        return x, y




def get_most_likely_row(tokens, mask, logits):
    # evaluate the autoregressive loss at all positions
    shift_logits = (logits[..., :-1, :]).contiguous()
    shift_tokens = (tokens[..., 1:]).contiguous()
    flat_shift_logits = shift_logits.view(-1, shift_logits.size(-1))
    flat_shift_tokens = shift_tokens.view(-1)
    shift_losses = F.cross_entropy(flat_shift_logits, flat_shift_tokens, reduction='none')
    shift_losses = shift_losses.view(tokens.size(0), -1)
    # now get the average loss just for the completion region (where mask == 1), in each row
    shift_mask = (mask[..., 1:]).contiguous() # we must shift mask, so we start at the last prompt token
    masked_shift_losses = shift_losses * shift_mask
    # sum and divide by the number of 1s in the mask
    sum_loss = masked_shift_losses.sum(dim=1)
    avg_loss = sum_loss / shift_mask.sum(dim=1)
    # now we have a loss for each of the 4 completions
    # the one with the lowest loss should be the most likely
    pred_norm = avg_loss.argmin().item()
    return pred_norm
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




# DDP launch for e.g. 8 GPUs:
# torchrun --standalone --nproc_per_node=8 train_gpt2.py
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import os
# set up DDP (distributed data parallel).
# torchrun command sets the env variables RANK, LOCAL_RANK, and WORLD_SIZE
ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
if ddp:
    # use of DDP atm demands CUDA, we set the device appropriately according to rank
    assert torch.cuda.is_available(), "for now i think we need CUDA for DDP"
    init_process_group(backend='nccl')
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
else:
    # vanilla, non-DDP run
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True
    # attempt to autodetect device
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    print(f"using device: {device}")

# added after video, pytorch can be serious about it's device vs. device_type distinction
device_type = "cuda" if device.startswith("cuda") else "cpu"



torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.manual_seed(1337)
enc = tiktoken.get_encoding("gpt2")

total_batch_size = 32768 #524288 for a larger GPU
B = 4 # 16 for a larger GPU
T = 1024 
assert total_batch_size % (B * T * ddp_world_size) == 0
grad_accum_steps = total_batch_size // (B * T * ddp_world_size)
if master_process:
    print(f"Total desired batch size: {total_batch_size}")
    print(f"=> Calculated gradient accumulation steps: {grad_accum_steps}")

train_loader = Dataloader(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size,split='train')
val_loader = Dataloader(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size,split='val')

torch.set_float32_matmul_precision('high')

num_return_sequences = 5
max_length = 30

max_lr = 6e-4
max_steps = 19073
warmup_step = 715
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



model = GPT(GPTConfig(vocab_size=50304))
model.to(device)
use_compile = False # torch.compile interferes with HellaSwag eval and Generation.
if use_compile:
    model = torch.compile(model)
if ddp:
    model = DDP(model,device_ids=[ddp_local_rank])
raw_model = model.module if ddp else model
# model = torch.compile(model) # if you are using A100 and above


#optimize our losses
# optimizer = torch.optim.AdamW(model.parameters(),lr=3e-4, betas=(0.9,0.95),eps=1e-8)
optimizer = raw_model.configure_optimizers(weight_decay=0.1,learning_rate=6e-4,device_type=device)



# create the log directory we will write checkpoints to and log to
log_dir = "log"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"log.txt")
with open(log_file, "w") as f: # open for writing to clear the file
    pass

for step in range(max_steps):
    t0 = time.time()
    last_step = (step == max_steps - 1)

    # once in a while evaluate our validation loss
    if step % 250 == 0 or last_step:
        model.eval()
        val_loader.reset()
        with torch.no_grad():
            val_loss_accum = 0.0
            val_loss_steps = 20
            for _ in range(val_loss_steps):
                x, y = val_loader.next_batch()
                x, y = x.to(device), y.to(device)
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                    logits, loss = model(x, y)
                loss = loss / val_loss_steps
                val_loss_accum += loss.detach()
        if ddp:
            dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)
        if master_process:
            print(f"validation loss: {val_loss_accum.item():.4f}")
            with open(log_file, "a") as f:
                f.write(f"{step} val {val_loss_accum.item():.4f}\n")
            if step > 0 and (step % 5000 == 0 or last_step):
                # optionally write model checkpoints
                checkpoint_path = os.path.join(log_dir, f"model_{step:05d}.pt")
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'config': raw_model.config,
                    'step': step,
                    'val_loss': val_loss_accum.item()
                }
                # you might also want to add optimizer.state_dict() and
                # rng seeds etc., if you wanted to more exactly resume training
                torch.save(checkpoint, checkpoint_path)


# once in a while evaluate hellaswag
    if (step % 250 == 0 or last_step) and (not use_compile):
        num_correct_norm = 0
        num_total = 0
        for i, example in enumerate(iterate_examples("val")):
            # only process examples where i % ddp_world_size == ddp_rank
            if i % ddp_world_size != ddp_rank:
                continue
            # render the example into tokens and labels
            _, tokens, mask, label = render_example(example)
            tokens = tokens.to(device)
            mask = mask.to(device)
            # get the logits
            with torch.no_grad():
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                    logits, loss = model(tokens)
                pred_norm = get_most_likely_row(tokens, mask, logits)
            num_total += 1
            num_correct_norm += int(pred_norm == label)
        # reduce the stats across all processes
        if ddp:
            num_total = torch.tensor(num_total, dtype=torch.long, device=device)
            num_correct_norm = torch.tensor(num_correct_norm, dtype=torch.long, device=device)
            dist.all_reduce(num_total, op=dist.ReduceOp.SUM)
            dist.all_reduce(num_correct_norm, op=dist.ReduceOp.SUM)
            num_total = num_total.item()
            num_correct_norm = num_correct_norm.item()
        acc_norm = num_correct_norm / num_total
        if master_process:
            print(f"HellaSwag accuracy: {num_correct_norm}/{num_total}={acc_norm:.4f}")
            with open(log_file, "a") as f:
                f.write(f"{step} hella {acc_norm:.4f}\n")

# once in a while generate from the model (except step 0, which is noise)
    if ((step > 0 and step % 250 == 0) or last_step) and (not use_compile):
        model.eval()
        num_return_sequences = 4
        max_length = 32
        tokens = enc.encode("Hello, I'm a language model,")
        tokens = torch.tensor(tokens, dtype=torch.long)
        tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
        xgen = tokens.to(device)
        sample_rng = torch.Generator(device=device)
        sample_rng.manual_seed(42 + ddp_rank)
        while xgen.size(1) < max_length:
            # forward the model to get the logits
            with torch.no_grad():
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                    logits, loss = model(xgen) # (B, T, vocab_size)
                # take the logits at the last position
                logits = logits[:, -1, :] # (B, vocab_size)
                # get the probabilities
                probs = F.softmax(logits, dim=-1)
                topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
                ix = torch.multinomial(topk_probs, 1, generator=sample_rng) # (B, 1)
                xcol = torch.gather(topk_indices, -1, ix) # (B, 1)
                xgen = torch.cat((xgen, xcol), dim=1)
        # print the generated text
        for i in range(num_return_sequences):
            tokens = xgen[i, :max_length].tolist()
            decoded = enc.decode(tokens)
            print(f"rank {ddp_rank} sample {i}: {decoded}")

    # do one step of the optimization
    model.train()
    optimizer.zero_grad()
    loss_accum = 0.0
    for micro_step in range(grad_accum_steps):
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)
        # added after video, this field is also used by the forward pass.
        if ddp:
            model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
            logits, loss = model(x, y)
        # we have to scale the loss to account for gradient accumulation,
        # because the gradients just add on each successive backward().
        # addition of gradients corresponds to a SUM in the objective, but
        # instead of a SUM we want MEAN. Scale the loss here so it comes out right
        loss = loss / grad_accum_steps
        loss_accum += loss.detach()
        loss.backward()
    if ddp:
        dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    # determine and set the learning rate for this iteration
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    optimizer.step()
    if device_type == "cuda":
        torch.cuda.synchronize() # wait for the GPU to finish work
    t1 = time.time()
    dt = t1 - t0 # time difference in seconds
    tokens_processed = train_loader.B * train_loader.T * grad_accum_steps * ddp_world_size
    tokens_per_sec = tokens_processed / dt
    if master_process:
        print(f"step {step:5d} | loss: {loss_accum.item():.6f} | lr {lr:.4e} | norm: {norm:.4f} | dt: {dt*1000:.2f}ms | tok/sec: {tokens_per_sec:.2f}")
        with open(log_file, "a") as f:
            f.write(f"{step} train {loss_accum.item():.6f}\n")

if ddp:
    destroy_process_group()