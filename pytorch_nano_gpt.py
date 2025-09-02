import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
from utilities import read_file_from, Token_Translator


class Causal_self_Attention(nn.Module):
    def __init__(self, embed_size, num_heads=4):
        super(Causal_self_Attention, self).__init__()
        self.embed_size = embed_size
        self.num_heads = num_heads
        self.head_dim = embed_size // num_heads
        if embed_size % num_heads != 0:
            raise AssertionError("Embedding size must be divisible by number of heads")

        self.query = nn.Linear(embed_size, embed_size)
        self.key = nn.Linear(embed_size, embed_size)
        self.value = nn.Linear(embed_size, embed_size)

        self.out_proj = nn.Linear(embed_size, embed_size)

        self.register_buffer("mask", None, persistent=False)

    def forward(self, x):
        batch_size, seq_len, embed_size = x.size()

        if self.mask is None or self.mask.size(0) != seq_len:
            mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
            self.register_buffer("mask", mask, persistent=False)
       

        Q_proj = self.query(x)
        K_proj = self.key(x)
        V_proj = self.value(x)

        Q_heads = Q_proj.reshape(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        K_heads = K_proj.reshape(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        V_heads = V_proj.reshape(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        attn_scores = torch.einsum('bhqd,bhkd->bhqk', Q_heads, K_heads) / math.sqrt(self.head_dim)
        attn_scores = attn_scores.masked_fill(self.mask, float('-inf'))

        attn_prob = F.softmax(attn_scores, dim=-1)

        weighted_value = torch.einsum('bhqk,bhkd->bhqd', attn_prob, V_heads)

        concat = weighted_value.permute(0, 2, 1, 3).contiguous().reshape(batch_size, seq_len, embed_size)

        output = self.out_proj(concat)

        return output


class Feed_Forward(nn.Module):
    def __init__(self, embed_size, hidden_size=None):
        super(Feed_Forward, self).__init__()
        hidden_size = hidden_size or 4 * embed_size
        self.fc1 = nn.Linear(embed_size, hidden_size)
        self.gelu = nn.GELU()
        self.fc2 = nn.Linear(hidden_size, embed_size)

    def forward(self, x):
        intermediate = self.fc1(x)
        activated = self.gelu(intermediate)
        output = self.fc2(activated)
        return output


class Transformer_Block(nn.Module):
    def __init__(self, embed_size, num_heads):
        super(Transformer_Block, self).__init__()
        self.norm1 = nn.LayerNorm(embed_size)
        self.self_attention = Causal_self_Attention(embed_size, num_heads)
        self.norm2 = nn.LayerNorm(embed_size)
        self.feed_forward = Feed_Forward(embed_size)

    def forward(self, x):
        x_normed1 = self.norm1(x)
        attention_out = self.self_attention(x_normed1)
        x_residual1 = x + attention_out
        x_normed2 = self.norm2(x_residual1)
        ff_out = self.feed_forward(x_normed2)
        x_residual2 = x_residual1 + ff_out
        return x_residual2


class NanoGPT(nn.Module):
    def __init__(self, vocab_size, embed_size, max_seq_len, num_layers=4, num_heads=4):
        super(NanoGPT, self).__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.max_seq_len = max_seq_len

        self.token_embedding = nn.Embedding(vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_seq_len, embed_size)

        self.layers = nn.ModuleList([
            Transformer_Block(embed_size, num_heads) for _ in range(num_layers)
        ])

        self.layer_norm = nn.LayerNorm(embed_size)
        self.output_linear = nn.Linear(embed_size, vocab_size, bias=False)

    def forward(self, input_tokens):
        batch_size, seq_len = input_tokens.size()
        assert seq_len <= self.max_seq_len, f"Input too long ({seq_len} > {self.max_seq_len})"

        token_embeds = self.token_embedding(input_tokens)
        positions = torch.arange(seq_len, device=input_tokens.device).unsqueeze(0).expand(batch_size, seq_len)
        pos_embeds = self.position_embedding(positions)

        x = token_embeds + pos_embeds

        for layer in self.layers:
            x = layer(x)

        x = self.layer_norm(x)
        logits = self.output_linear(x)
        return logits

    @torch.no_grad()
    def generate(self, prefix_tokens, max_new_tokens, temperature=1.0, top_k=None):
        generated = prefix_tokens.clone()
        for _ in range(max_new_tokens):
            input_ids = generated[:, -self.max_seq_len:] if generated.size(1) > self.max_seq_len else generated
            logits = self(input_ids)[:, -1, :] / temperature

            if top_k is not None:
                top_values, _ = torch.topk(logits, top_k)
                logits[logits < top_values[:, [-1]]] = float('-inf')

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, next_token], dim=1)
        return generated

    def get_perplexity(self, tokens, targets=None):
        if targets is None:
            targets = tokens[:, 1:]
            tokens = tokens[:, :-1]
        logits = self(tokens)

        logits_flat = logits.reshape(-1, self.vocab_size)
        targets_flat = targets.reshape(-1)

        loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(logits_flat, targets_flat)
        return torch.exp(loss)


def create_batches(token_sequence, batch_length):
    full_batches = len(token_sequence) // batch_length
    trimmed_seq = token_sequence[:full_batches * batch_length]
    return torch.tensor(trimmed_seq, dtype=torch.long).view(full_batches, batch_length)


K = 400
embed_dim = 64
max_context = 128
layers = 3
heads = 4

vocabulary = read_file_from("train", "learned_vocabularies", K)
corpus_tokens = read_file_from("train", "tokenized_corpus", K)
vocab_size = len(vocabulary)

token_translator = Token_Translator(vocabulary)
encoded_tokens = token_translator.encode_list(corpus_tokens)

inputs = encoded_tokens[:-1]
targets = encoded_tokens[1:]

input_batches = create_batches(inputs, max_context)
target_batches = create_batches(targets, max_context)

model = NanoGPT(vocab_size, embed_dim, max_context, layers, heads)
model.eval()

"""
for i in range(input_batches.size(0)):
    batch_input = input_batches[i].unsqueeze(0)
    batch_target = target_batches[i].unsqueeze(0)
    perp = model.get_perplexity(batch_input, targets=batch_target)
    print(f"Current batch perplexity: {perp:.3f}")
"""

prefix = input_batches[0, :5].unsqueeze(0)
generated = model.generate(prefix, max_new_tokens=50, temperature=1.0, top_k=3)
generated_list = generated[0].tolist()
generated_text = "".join(token_translator.decode_list(generated_list))
print("Generated text:", generated_text.replace("</w>", " "))



optimizer = optim.Adam(model.parameters(), lr=1e-4)  # adjust lr as needed

# Define loss criterion
criterion = nn.CrossEntropyLoss()

# Training loop example (simplified):
num_epochs = 5
model.train()
for epoch in range(num_epochs):
    for i in range(input_batches.size(0)):
        batch_input = input_batches[i].unsqueeze(0)
        batch_target = target_batches[i].unsqueeze(0)

        optimizer.zero_grad()             # Clear gradients
        logits = model(batch_input)       # Forward pass
        loss = criterion(
            logits.view(-1, vocab_size),  # reshape logits for loss
            batch_target.view(-1)
        )
        loss.backward()                   # Backpropagation
        optimizer.step()                  # Update weights

        print(f"Epoch {epoch}, Batch {i}, Loss: {loss.item():.4f}")

prefix = input_batches[0, :5].unsqueeze(0)
generated = model.generate(prefix, max_new_tokens=50, temperature=1.0, top_k=3)
generated_list = generated[0].tolist()
generated_text = "".join(token_translator.decode_list(generated_list))
print("Generated text:", generated_text.replace("</w>", " "))