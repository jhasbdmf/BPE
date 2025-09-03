import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
import copy
from utilities import read_file_from, Token_Translator, log_message
import datetime


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

    def get_perplexity(self, tokens, train_targets=None):
        if train_targets is None:
            train_targets = tokens[:, 1:]
            tokens = tokens[:, :-1]
        logits = self(tokens)

        logits_flat = logits.reshape(-1, self.vocab_size)
        train_targets_flat = train_targets.reshape(-1)

        loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(logits_flat, train_targets_flat)
        return torch.exp(loss)


def create_batches(token_sequence, batch_length):
    full_batches = len(token_sequence) // batch_length
    trimmed_seq = token_sequence[:full_batches * batch_length]
    return torch.tensor(trimmed_seq, dtype=torch.long).view(full_batches, batch_length)



def train_model(model, 
                train_input_batches, 
                train_target_batches, 
                val_input_batches, 
                val_target_batches, 
                num_epochs=5, 
                lr=1e-4, 
                seed=None, 
                patience=3, 
                verbose=True):
    if seed is not None:
        torch.manual_seed(seed)
    #optimizer = optim.Adam(model.parameters(), lr=lr)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float('inf')
    epochs_without_improvement = 0

    num_batches = train_input_batches.size(0)

    train_loss_history = []
    val_loss_history = []
    best_model_state = copy.deepcopy(model.state_dict())

    for epoch in range(num_epochs):
        if seed is not None:
            torch.manual_seed(seed + epoch)

        perm = torch.randperm(num_batches)
        train_input_batches_shuffled = train_input_batches[perm]
        train_target_batches_shuffled = train_target_batches[perm]

        model.train()
        train_loss_total = 0.0
        for i in range(num_batches):
            batch_input = train_input_batches_shuffled[i].unsqueeze(0)
            batch_target = train_target_batches_shuffled[i].unsqueeze(0)

            optimizer.zero_grad()
            logits = model(batch_input)
            loss = criterion(logits.view(-1, model.vocab_size), batch_target.view(-1))
            loss.backward()
            optimizer.step()

            train_loss_total += loss.item()

            #if verbose:
            #    print(f"Epoch {epoch+1}/{num_epochs}, Batch {i+1}/{num_batches}, Loss: {loss.item():.4f}")
            #    log_message(f"Epoch {epoch+1}/{num_epochs}, Batch {i+1}/{num_batches}, Loss: {loss.item():.4f}", filename)

        avg_train_loss = train_loss_total / num_batches
        train_loss_history.append(avg_train_loss)

        model.eval()
        val_loss_total = 0.0
        with torch.no_grad():
            for j in range(val_input_batches.size(0)):
                val_input = val_input_batches[j].unsqueeze(0)
                val_target = val_target_batches[j].unsqueeze(0)
                val_logits = model(val_input)
                val_loss = criterion(val_logits.view(-1, model.vocab_size), val_target.view(-1))
                val_loss_total += val_loss.item()

        avg_val_loss = val_loss_total / val_input_batches.size(0)
        val_loss_history.append(avg_val_loss)

        if verbose:
            print(f"Epoch {epoch+1}/{num_epochs} completed.\n Avg Train Loss: {avg_train_loss:.4f},\n Avg Val Loss: {avg_val_loss:.4f}")
            print ("_" * 50)
            log_message(f"Epoch {epoch+1}/{num_epochs} completed.\n Avg Train Loss: {avg_train_loss:.4f},\n Avg Val Loss: {avg_val_loss:.4f}", filename)
            log_message ("_" * 50, filename)


        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if verbose:
                print(f"No improvement for {epochs_without_improvement} epochs")
                log_message(f"No improvement for {epochs_without_improvement} epochs", filename)
            if epochs_without_improvement >= patience:
                if verbose:
                    print(f"Early stopping triggered after {epoch+1} epochs")
                    log_message(f"Early stopping triggered after {epoch+1} epochs", filename)
                break

    model.load_state_dict(best_model_state)
    return model, train_loss_history, val_loss_history




def grid_search(hyperparameters: dict,
                train_tokens_raw: dict,  # dict mapping K -> raw tokens list
                val_tokens_raw: dict,    # likewise for validation tokens
                n_epochs: int,
                batch_seq_len_default=128):
    best_val_loss = float("inf")
    best_params = {}
    best_model = None
    best_train_loss_history = None
    best_val_loss_history = None

    Ks = hyperparameters.get('vocab_size', [400])  # Add vocab size hyperparameter

    learning_rates = hyperparameters.get('lr', [1e-4])
    num_layers_list = hyperparameters.get('n_layers', [4])
    embed_dims = hyperparameters.get('embed_dim', [64])
    num_heads_list = hyperparameters.get('num_heads', [4])
    max_ctxs = hyperparameters.get('max_context', [batch_seq_len_default])

    for K in Ks:
        print ("_" * 100)
        print(f"Using vocabulary size K={K}")
        

        # Load or prepare vocabulary & token translator for current K
        vocabulary = read_file_from("train", "learned_vocabularies", K)
        token_translator = Token_Translator(vocabulary)

        # Get raw tokens for current K
        train_tokens = train_tokens_raw[K]
        val_tokens = val_tokens_raw[K]

        # Encode tokens, prepare batches outside inner hyperparam loops for efficiency
        train_encoded = token_translator.encode_list(train_tokens)
        train_inputs = train_encoded[:-1]
        train_targets = train_encoded[1:]

        val_encoded = token_translator.encode_list(val_tokens)
        val_inputs = val_encoded[:-1]
        val_targets = val_encoded[1:]

        for lr in learning_rates:
            for n_layers in num_layers_list:
                for embed_dim in embed_dims:
                    for n_heads in num_heads_list:
                        for max_ctx in max_ctxs:
                            print(f"Testing: K={K}, lr={lr}, layers={n_layers}, embed_dim={embed_dim}, heads={n_heads}, max_context={max_ctx}")
                            print ("_" * 75)

                            log_message ("_" * 100, filename)
                            log_message(f"Testing: K={K}, lr={lr}, layers={n_layers}, embed_dim={embed_dim}, heads={n_heads}, max_context={max_ctx}", filename)
                            log_message ("_" * 75, filename)
                            
                            train_input_batches = create_batches(train_inputs, max_ctx)
                            train_target_batches = create_batches(train_targets, max_ctx)

                            val_input_batches = create_batches(val_inputs, max_ctx)
                            val_target_batches = create_batches(val_targets, max_ctx)

                            model = NanoGPT(vocab_size=len(vocabulary), 
                                           embed_size=embed_dim, 
                                           max_seq_len=max_ctx, 
                                           num_layers=n_layers, 
                                           num_heads=n_heads)

                            trained_model, train_loss_hist, val_loss_hist = train_model(
                                model,
                                train_input_batches,
                                train_target_batches,
                                val_input_batches,
                                val_target_batches,
                                num_epochs=n_epochs,
                                lr=lr,
                                patience=3,
                                verbose=True
                            )

                            min_val_loss = min(val_loss_hist)
                            if min_val_loss < best_val_loss:
                                best_val_loss = min_val_loss
                                
                                best_params = {
                                    'vocab_size': K,
                                    'learning_rate': lr,
                                    'num_layers': n_layers,
                                    'embed_dim': embed_dim,
                                    'num_heads': n_heads,
                                    'max_context': max_ctx
                                }
                                best_model = trained_model
                                best_train_loss_history = train_loss_hist
                                best_val_loss_history = val_loss_hist

                            print(f"Current best val loss: {best_val_loss:.5f}")
                            #log_message("_" * 50, filename)
                            log_message (f"Current best val loss: {best_val_loss:.5f}", filename)

    return best_train_loss_history, best_val_loss_history, best_model, best_params, best_val_loss


#name of log file to save training progress there
filename = f"training_log_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"


# Define K values to test
K_values = [400, 1200, 2000]

# Prepare dictionaries mapping K -> token lists (raw)
train_tokens_raw = {}
val_tokens_raw = {}

for K in K_values:
    train_tokens_raw[K] = read_file_from("train", "tokenized_corpus", K)
    val_tokens_raw[K] = read_file_from("valid", "tokenized_corpus", K)

# Define other hyperparameters to search over
hyperparams = {
    'vocab_size': K_values,
    'lr': [1e-3],
    'n_layers': [4],
    'embed_dim': [32],
    'num_heads': [4],
    'max_context': [128]
}

# Call grid_search
best_grid_search_train_loss_hist, best_grid_search_val_loss_hist, best_model, best_params, best_val_loss = grid_search(
    hyperparameters=hyperparams,
    train_tokens_raw=train_tokens_raw,
    val_tokens_raw=val_tokens_raw,
    #vocab=None,  
    n_epochs=1
)


vocabulary = read_file_from("train", "learned_vocabularies", best_params['vocab_size'])
token_translator = Token_Translator(vocabulary)
train_tokens = read_file_from("train", "tokenized_corpus", best_params['vocab_size'])

train_encoded = token_translator.encode_list(train_tokens)
train_inputs = train_encoded[:-1]
train_targets = train_encoded[1:]

# Use the max_seq_len from best_model for batching consistency
train_input_batches = create_batches(train_inputs, best_model.max_seq_len)
train_target_batches = create_batches(train_targets, best_model.max_seq_len)

# Take prefix from flat encoded tokens, before batching
prefix_encoded = train_encoded[:5]
prefix = torch.tensor([prefix_encoded], dtype=torch.long)

# Generate from prefix tensor
generated = best_model.generate(prefix, max_new_tokens=50, temperature=1.0, top_k=3)
generated_list = generated[0].tolist()
generated_text = "".join(token_translator.decode_list(generated_list))
generated_text = generated_text.replace("</w>", " ")
print("Generated text:", generated_text)

log_message("_" * 100, filename)
log_message (f"Generated text after grid search:\n {generated_text}", filename)
log_message("_" * 100, filename)




#########################################################################

# !!!train the best model further
vocabulary = read_file_from("train", "learned_vocabularies", best_params['vocab_size'])
token_translator = Token_Translator(vocabulary)

# Load training tokens for best vocab size
train_tokens = read_file_from("train", "tokenized_corpus", best_params['vocab_size'])
train_encoded = token_translator.encode_list(train_tokens)
train_inputs = train_encoded[:-1]
train_targets = train_encoded[1:]

# Create training batches using best model's max sequence length
train_input_batches = create_batches(train_inputs, best_model.max_seq_len)
train_target_batches = create_batches(train_targets, best_model.max_seq_len)

# Load validation tokens and create validation batches similarly
val_tokens = read_file_from("valid", "tokenized_corpus", best_params['vocab_size'])
val_encoded = token_translator.encode_list(val_tokens)
val_inputs = val_encoded[:-1]
val_targets = val_encoded[1:]
val_input_batches = create_batches(val_inputs, best_model.max_seq_len)
val_target_batches = create_batches(val_targets, best_model.max_seq_len)

# Optionally, continue training the best model for more epochs
additional_epochs = 2  # Set desired additional epochs
learning_rate = best_params['learning_rate']

log_message("Training best grid search model", filename)
log_message("_" * 75, filename)

best_model, train_loss_hist, val_loss_hist = train_model(
    best_model,
    train_input_batches,
    train_target_batches,
    val_input_batches,
    val_target_batches,
    num_epochs=additional_epochs,
    lr=learning_rate,
    patience=3,
    verbose=True,
    seed=None
)

print("Further training complete.")

# Prepare prefix tokens for generation from the start of training data
prefix_encoded = train_encoded[:5]  # first 5 tokens as prefix
prefix = torch.tensor([prefix_encoded], dtype=torch.long)

# Generate text from the trained model
generated = best_model.generate(prefix, max_new_tokens=50, temperature=1.0, top_k=3)
generated_list = generated[0].tolist()
generated_text = "".join(token_translator.decode_list(generated_list))
generated_text = generated_text.replace("</w>", " ")

# Replace end-of-word tokens with spaces if desired
print("Generated text:", generated_text)
log_message(f"Generated text after grid search and specicif best model training:\n {generated_text}", filename)
log_message("_" * 100, filename)