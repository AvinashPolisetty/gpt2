num_repeat_sequences = 5
max_length = 30

#model = GPT.from_pretrained('gpt2')      # for directly loading the weights from the huggingface
model = GPT(GPTConfig())                  # initalize the model from the strach randomly
model.eval()
model.to(device)


#prefix tokens encoding using tiktoken lib
import tiktoken
enc = tiktoken.get_encoding('gpt2')
tokens = enc.encode("Hello, I'm a language model,")
tokens = torch.tensor(tokens,dtype=torch.long)
tokens = tokens.unsqueeze(0).repeat(num_repeat_sequences,1)
x = tokens.to(device)

torch.manual_seed(42)

#sampling from the probs
while x.size(1) < max_length:
    with torch.no_grad():

        logits = model(x)
        # take the logits at the last position
        logits = logits[:, -1, :] # (B, vocab_size)
        # get the probabilities
        probs = F.softmax(logits, dim=-1)
        # do top-k sampling of 50 (huggingface pipeline default)
        # topk_probs here becomes (5, 50), topk_indices is (5, 50)
        topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
        # select a token from the top-k probabilities
        # note: multinomial does not demand the input to sum to 1
        ix = torch.multinomial(topk_probs, 1) # (B, 1)
        # gather the corresponding indices
        xcol = torch.gather(topk_indices, -1, ix) # (B, 1)
        #append to the sequence
        x = torch.cat((x,xcol),dim=1)


#print the generated text
for i in range(num_repeat_sequences):
    tokens = x[i,:max_length].tolist()
    decoded = enc.decode(tokens)
    print(">",decoded)
