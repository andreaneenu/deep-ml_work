import numpy as np

def gen_text(prompt: str, n_tokens_to_generate: int = 40):
	# Your code here
	encoder, hparams, params = load_encoder_hparams_and_params()
	tokens = encoder.encode(prompt)
	initial_tokens = len(tokens)
	for i in range(n_tokens_to_generate):
		seq_len = len(tokens)
		x = params["wte"][tokens] + params["wpe"][:seq_len]
		# x = transformer_block(x, hparams, params)
		for i in range(len(x)):
			x[i] = layernorm(x[i], params["ln_f"]["g"], params["ln_f"]["b"])
		logits = x @ params["wte"].T
		next_token = int(np.argmax(logits[-1]))
		tokens.append(next_token)
	tokens = tokens[initial_tokens:]
	return encoder.decode(tokens)


def layernorm(x, g, b, epsilon=1e-5):
	return g*((x-np.mean(x))/np.sqrt(np.var(x)+epsilon))+b


def load_encoder_hparams_and_params(model_size: str = "124M", models_dir: str = "models"):
	class DummyBPE:
		def __init__(self):
			self.encoder_dict = {"hello": 1, "world": 2, "<UNK>": 0}

		def encode(self, text: str):
			tokens = text.strip().split()
			return [self.encoder_dict.get(token, self.encoder_dict["<UNK>"]) for token in tokens]

		def decode(self, token_ids: list):
			reversed_dict = {v: k for k, v in self.encoder_dict.items()}
			return " ".join([reversed_dict.get(tok_id, "<UNK>") for tok_id in token_ids])

	hparams = {
		"n_ctx": 1024,
		"n_head": 2
	}

	params = {
		"wte": np.random.rand(3, 10),
		"wpe": np.random.rand(1024, 10),
		"blocks": [],
		"ln_f": {
			"g": np.ones(10),
			"b": np.zeros(10),
		}
	}

	encoder = DummyBPE()
	return encoder, hparams, params
