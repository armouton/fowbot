"""
Neural network models for next-word prediction (NumPy only).
Supports two architectures: feedforward and LSTM.
"""

import numpy as np
import pickle
import time
from collections import Counter
from typing import List, Tuple, Optional, Callable


def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


class SimpleNextWordModel:
    """Feedforward neural network for next-word prediction."""

    def __init__(self, vocab_size: int, context_length: int, hidden_size: int = 128):
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.hidden_size = hidden_size

        self.W_embed = np.random.randn(vocab_size, hidden_size) * 0.01
        self.W_hidden = np.random.randn(context_length * hidden_size, hidden_size) * 0.01
        self.b_hidden = np.zeros(hidden_size)
        self.W_output = np.random.randn(hidden_size, vocab_size) * 0.01
        self.b_output = np.zeros(vocab_size)

    def forward(self, x):
        batch_size = x.shape[0]
        embedded = self.W_embed[x]
        flattened = embedded.reshape(batch_size, -1)
        hidden = np.maximum(0, flattened @ self.W_hidden + self.b_hidden)
        logits = hidden @ self.W_output + self.b_output
        return logits, hidden, embedded

    def predict_proba(self, x):
        logits, _, _ = self.forward(x)
        return softmax(logits)

    def param_count(self):
        return (self.W_embed.size + self.W_hidden.size + self.b_hidden.size +
                self.W_output.size + self.b_output.size)


class LSTMNextWordModel:
    """LSTM neural network for next-word prediction."""

    def __init__(self, vocab_size: int, hidden_size: int = 128):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        H = hidden_size

        # Embedding layer (shared with feedforward concept)
        self.W_embed = np.random.randn(vocab_size, H) * 0.01

        # LSTM gate weights: input is [h_prev, x_embed] of size 2*H
        # Using Xavier initialization for better gradient flow
        scale = np.sqrt(2.0 / (2 * H))
        self.W_f = np.random.randn(2 * H, H) * scale  # forget gate
        self.b_f = np.ones(H)  # bias toward remembering (forget gate starts open)
        self.W_i = np.random.randn(2 * H, H) * scale   # input gate
        self.b_i = np.zeros(H)
        self.W_o = np.random.randn(2 * H, H) * scale   # output gate
        self.b_o = np.zeros(H)
        self.W_c = np.random.randn(2 * H, H) * scale   # candidate cell
        self.b_c = np.zeros(H)

        # Output projection
        self.W_output = np.random.randn(H, vocab_size) * 0.01
        self.b_output = np.zeros(vocab_size)

    def forward_step(self, x_embed, h_prev, c_prev):
        """Single LSTM timestep. x_embed: (batch, H), h_prev: (batch, H), c_prev: (batch, H)."""
        hx = np.concatenate([h_prev, x_embed], axis=1)  # (batch, 2H)

        f = sigmoid(hx @ self.W_f + self.b_f)
        i = sigmoid(hx @ self.W_i + self.b_i)
        o = sigmoid(hx @ self.W_o + self.b_o)
        c_hat = np.tanh(hx @ self.W_c + self.b_c)

        c = f * c_prev + i * c_hat
        h = o * np.tanh(c)

        cache = (hx, f, i, o, c_hat, c_prev, c, h_prev, x_embed)
        return h, c, cache

    def forward_sequence(self, x_indices):
        """Forward pass over a sequence. x_indices: (batch, seq_len) integer array."""
        batch_size, seq_len = x_indices.shape
        H = self.hidden_size

        h = np.zeros((batch_size, H))
        c = np.zeros((batch_size, H))
        caches = []

        for t in range(seq_len):
            x_embed = self.W_embed[x_indices[:, t]]  # (batch, H)
            h, c, cache = self.forward_step(x_embed, h, c)
            caches.append(cache)

        logits = h @ self.W_output + self.b_output
        return logits, h, c, caches

    def predict_proba(self, x_indices):
        logits, _, _, _ = self.forward_sequence(x_indices)
        return softmax(logits)

    def param_count(self):
        return (self.W_embed.size +
                self.W_f.size + self.b_f.size +
                self.W_i.size + self.b_i.size +
                self.W_o.size + self.b_o.size +
                self.W_c.size + self.b_c.size +
                self.W_output.size + self.b_output.size)


class WordPredictor:
    """Main class for training and using the next-word prediction model."""

    def __init__(self, context_length: int = 5, vocab_size: int = 5000):
        self.context_length = context_length
        self.max_vocab_size = vocab_size
        self.word2idx = {}
        self.idx2word = {}
        self.model = None
        self.architecture = "feedforward"
        self._stop_requested = False

    def build_vocabulary(self, text: str) -> int:
        words = text.lower().split()
        word_counts = Counter(words)
        most_common = word_counts.most_common(self.max_vocab_size - 2)

        self.word2idx = {'<UNK>': 0, '<PAD>': 1}
        self.idx2word = {0: '<UNK>', 1: '<PAD>'}

        for idx, (word, _) in enumerate(most_common, start=2):
            self.word2idx[word] = idx
            self.idx2word[idx] = word

        return len(self.word2idx)

    def prepare_training_data(self, text: str) -> Tuple[np.ndarray, np.ndarray]:
        words = text.lower().split()
        sequences = []
        targets = []

        for i in range(len(words) - self.context_length):
            context = words[i:i + self.context_length]
            target = words[i + self.context_length]

            context_indices = [self.word2idx.get(w, 0) for w in context]
            target_idx = self.word2idx.get(target, 0)

            sequences.append(context_indices)
            targets.append(target_idx)

        X = np.array(sequences, dtype=np.int32)
        y = np.array(targets, dtype=np.int32)
        return X, y

    def train(self, text: str, epochs: int = 50, learning_rate: float = 1.0,
              batch_size: int = 64, hidden_size: int = 256,
              architecture: str = "feedforward",
              on_epoch: Optional[Callable[[int, int, float], None]] = None) -> None:
        self._stop_requested = False
        self.architecture = architecture

        vocab_size = self.build_vocabulary(text)
        X, y = self.prepare_training_data(text)

        if len(X) < batch_size:
            batch_size = max(1, len(X))

        if architecture == "lstm":
            self._train_lstm(X, y, vocab_size, epochs, learning_rate,
                             batch_size, hidden_size, on_epoch)
        else:
            self._train_feedforward(X, y, vocab_size, epochs, learning_rate,
                                    batch_size, hidden_size, on_epoch)

    def _train_feedforward(self, X, y, vocab_size, epochs, learning_rate,
                           batch_size, hidden_size, on_epoch):
        self.model = SimpleNextWordModel(vocab_size, self.context_length,
                                         hidden_size=hidden_size)
        n_batches = max(1, len(X) // batch_size)

        for epoch in range(epochs):
            if self._stop_requested:
                break

            indices = np.random.permutation(len(X))
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            epoch_loss = 0

            for batch in range(n_batches):
                batch_start = batch * batch_size
                batch_end = batch_start + batch_size
                X_batch = X_shuffled[batch_start:batch_end]
                y_batch = y_shuffled[batch_start:batch_end]
                actual_batch_size = len(X_batch)

                logits, hidden, embedded = self.model.forward(X_batch)
                probs = softmax(logits)

                batch_loss = -np.mean(np.log(probs[np.arange(actual_batch_size), y_batch] + 1e-10))
                epoch_loss += batch_loss

                d_logits = probs.copy()
                d_logits[np.arange(actual_batch_size), y_batch] -= 1
                d_logits /= actual_batch_size

                d_W_output = hidden.T @ d_logits
                d_b_output = np.sum(d_logits, axis=0)
                d_hidden = d_logits @ self.model.W_output.T
                d_hidden[hidden <= 0] = 0

                flattened = embedded.reshape(actual_batch_size, -1)
                d_W_hidden = flattened.T @ d_hidden
                d_b_hidden = np.sum(d_hidden, axis=0)
                d_flattened = d_hidden @ self.model.W_hidden.T
                d_embedded = d_flattened.reshape(actual_batch_size, self.context_length,
                                                 self.model.hidden_size)

                np.add.at(self.model.W_embed, X_batch, -learning_rate * d_embedded)
                self.model.W_output -= learning_rate * d_W_output
                self.model.b_output -= learning_rate * d_b_output
                self.model.W_hidden -= learning_rate * d_W_hidden
                self.model.b_hidden -= learning_rate * d_b_hidden

            avg_loss = epoch_loss / n_batches
            if on_epoch:
                on_epoch(epoch + 1, epochs, float(avg_loss))

    def _train_lstm(self, X, y, vocab_size, epochs, learning_rate,
                    batch_size, hidden_size, on_epoch):
        self.model = LSTMNextWordModel(vocab_size, hidden_size=hidden_size)
        m = self.model
        H = hidden_size
        n_batches = max(1, len(X) // batch_size)
        clip_norm = 5.0

        for epoch in range(epochs):
            if self._stop_requested:
                break

            indices = np.random.permutation(len(X))
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            epoch_loss = 0

            for batch in range(n_batches):
                batch_start = batch * batch_size
                batch_end = batch_start + batch_size
                X_batch = X_shuffled[batch_start:batch_end]
                y_batch = y_shuffled[batch_start:batch_end]
                bs = len(X_batch)
                seq_len = X_batch.shape[1]

                # Forward pass
                logits, h_final, c_final, caches = m.forward_sequence(X_batch)
                probs = softmax(logits)

                batch_loss = -np.mean(np.log(probs[np.arange(bs), y_batch] + 1e-10))
                epoch_loss += batch_loss

                # Output layer gradients
                d_logits = probs.copy()
                d_logits[np.arange(bs), y_batch] -= 1
                d_logits /= bs

                d_W_output = h_final.T @ d_logits
                d_b_output = np.sum(d_logits, axis=0)
                d_h = d_logits @ m.W_output.T  # (bs, H)
                d_c = np.zeros((bs, H))

                # Accumulate LSTM gate gradients
                d_W_f = np.zeros_like(m.W_f)
                d_b_f = np.zeros_like(m.b_f)
                d_W_i = np.zeros_like(m.W_i)
                d_b_i = np.zeros_like(m.b_i)
                d_W_o = np.zeros_like(m.W_o)
                d_b_o = np.zeros_like(m.b_o)
                d_W_c = np.zeros_like(m.W_c)
                d_b_c = np.zeros_like(m.b_c)
                d_embed_list = [None] * seq_len

                # BPTT: walk backward through timesteps
                for t in reversed(range(seq_len)):
                    hx, f, i, o, c_hat, c_prev, c_t, h_prev, x_embed = caches[t]

                    tanh_c = np.tanh(c_t)

                    # Gradient through h = o * tanh(c)
                    d_o = d_h * tanh_c
                    d_c += d_h * o * (1 - tanh_c ** 2)

                    # Gradient through c = f * c_prev + i * c_hat
                    d_f = d_c * c_prev
                    d_i = d_c * c_hat
                    d_c_hat = d_c * i
                    d_c_prev = d_c * f

                    # Gate activation gradients
                    d_f_raw = d_f * f * (1 - f)  # sigmoid derivative
                    d_i_raw = d_i * i * (1 - i)
                    d_o_raw = d_o * o * (1 - o)
                    d_c_hat_raw = d_c_hat * (1 - c_hat ** 2)  # tanh derivative

                    # Accumulate weight gradients
                    d_W_f += hx.T @ d_f_raw
                    d_b_f += np.sum(d_f_raw, axis=0)
                    d_W_i += hx.T @ d_i_raw
                    d_b_i += np.sum(d_i_raw, axis=0)
                    d_W_o += hx.T @ d_o_raw
                    d_b_o += np.sum(d_o_raw, axis=0)
                    d_W_c += hx.T @ d_c_hat_raw
                    d_b_c += np.sum(d_c_hat_raw, axis=0)

                    # Gradient w.r.t. concatenated input [h_prev, x_embed]
                    d_hx = (d_f_raw @ m.W_f.T + d_i_raw @ m.W_i.T +
                            d_o_raw @ m.W_o.T + d_c_hat_raw @ m.W_c.T)

                    d_h = d_hx[:, :H]  # gradient flowing to previous timestep's h
                    d_x_embed = d_hx[:, H:]  # gradient flowing to embedding
                    d_c = d_c_prev  # gradient flowing to previous timestep's c

                    d_embed_list[t] = d_x_embed

                # Gradient clipping by global norm
                all_grads = [d_W_f, d_b_f, d_W_i, d_b_i, d_W_o, d_b_o,
                             d_W_c, d_b_c, d_W_output, d_b_output]
                total_norm = np.sqrt(sum(np.sum(g ** 2) for g in all_grads))
                if total_norm > clip_norm:
                    scale = clip_norm / (total_norm + 1e-8)
                    for g in all_grads:
                        g *= scale

                # Apply updates
                m.W_output -= learning_rate * d_W_output
                m.b_output -= learning_rate * d_b_output
                m.W_f -= learning_rate * d_W_f
                m.b_f -= learning_rate * d_b_f
                m.W_i -= learning_rate * d_W_i
                m.b_i -= learning_rate * d_b_i
                m.W_o -= learning_rate * d_W_o
                m.b_o -= learning_rate * d_b_o
                m.W_c -= learning_rate * d_W_c
                m.b_c -= learning_rate * d_b_c

                # Embedding gradients
                for t in range(seq_len):
                    clip_scale = clip_norm / (total_norm + 1e-8) if total_norm > clip_norm else 1.0
                    np.add.at(m.W_embed, X_batch[:, t],
                              -learning_rate * clip_scale * d_embed_list[t])

            avg_loss = epoch_loss / n_batches
            if on_epoch:
                on_epoch(epoch + 1, epochs, float(avg_loss))

    def stop_training(self):
        self._stop_requested = True

    def predict_next_words(self, context: str, top_k: int = 5) -> List[Tuple[str, float]]:
        if self.model is None:
            raise ValueError("Model not trained yet!")

        words = context.lower().split()

        if len(words) > self.context_length:
            words = words[-self.context_length:]
        elif len(words) < self.context_length:
            words = ['<PAD>'] * (self.context_length - len(words)) + words

        indices = [self.word2idx.get(w, 0) for w in words]
        x = np.array([indices], dtype=np.int32)

        # Both architectures: process the context window from scratch
        probs = self.model.predict_proba(x)[0]

        n_candidates = min(top_k * 4, len(probs))
        top_indices = np.argsort(probs)[-n_candidates:][::-1]

        skip = {'<UNK>', '<PAD>'}
        predictions = []
        for idx in top_indices:
            word = self.idx2word[idx]
            if word in skip:
                continue
            predictions.append((word, float(probs[idx])))
            if len(predictions) >= top_k:
                break

        return predictions

    def generate_sequence(self, context: str, n_words: int = 1, top_k: int = 5) -> dict:
        """Generate n_words autoregressively, returning the sequence and per-step predictions."""
        if self.model is None:
            raise ValueError("Model not trained yet!")

        current_context = context
        generated_words = []
        steps = []

        for _ in range(n_words):
            predictions = self.predict_next_words(current_context, top_k=top_k)
            if not predictions:
                break
            best_word = predictions[0][0]
            generated_words.append(best_word)
            steps.append({
                "context": current_context,
                "predicted": best_word,
                "alternatives": [(w, float(p)) for w, p in predictions]
            })
            current_context = current_context + " " + best_word

        return {
            "original_context": context,
            "generated_text": context + " " + " ".join(generated_words),
            "generated_words": generated_words,
            "steps": steps,
        }

    def get_status(self) -> dict:
        params = 0
        if self.model is not None:
            params = self.model.param_count()
        return {
            "trained": self.model is not None,
            "vocab_size": len(self.word2idx) if self.word2idx else 0,
            "context_length": self.context_length,
            "parameters": params,
            "architecture": self.architecture,
        }

    def save(self, filepath: str) -> None:
        save_dict = {
            'model': self.model,
            'word2idx': self.word2idx,
            'idx2word': self.idx2word,
            'context_length': self.context_length,
            'max_vocab_size': self.max_vocab_size,
            'architecture': self.architecture,
        }
        with open(filepath, 'wb') as f:
            pickle.dump(save_dict, f)

    def load(self, filepath: str) -> None:
        with open(filepath, 'rb') as f:
            save_dict = pickle.load(f)
        self.model = save_dict['model']
        self.word2idx = save_dict['word2idx']
        self.idx2word = save_dict['idx2word']
        self.context_length = save_dict['context_length']
        self.max_vocab_size = save_dict['max_vocab_size']
        self.architecture = save_dict.get('architecture', 'feedforward')
