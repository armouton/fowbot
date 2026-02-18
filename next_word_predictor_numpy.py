#!/usr/bin/env python3
"""
Simple Next-Word Prediction Language Model (NumPy version)
Educational tool for understanding language model training
No external dependencies except NumPy
"""

import numpy as np
import pickle
import time
from collections import Counter
from typing import List, Tuple


class SimpleNextWordModel:
    """Simple neural network for next-word prediction using only NumPy"""
    
    def __init__(self, vocab_size: int, context_length: int, hidden_size: int = 128):
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.hidden_size = hidden_size
        
        # Initialize weights randomly
        # Embedding layer: vocab_size x hidden_size
        self.W_embed = np.random.randn(vocab_size, hidden_size) * 0.01
        
        # Hidden layer: (context_length * hidden_size) x hidden_size
        self.W_hidden = np.random.randn(context_length * hidden_size, hidden_size) * 0.01
        self.b_hidden = np.zeros(hidden_size)
        
        # Output layer: hidden_size x vocab_size
        self.W_output = np.random.randn(hidden_size, vocab_size) * 0.01
        self.b_output = np.zeros(vocab_size)
        
    def softmax(self, x):
        """Numerically stable softmax"""
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
    
    def relu(self, x):
        """ReLU activation"""
        return np.maximum(0, x)
    
    def forward(self, x):
        """Forward pass through the network"""
        batch_size = x.shape[0]
        
        # Embedding lookup
        embedded = self.W_embed[x]  # (batch_size, context_length, hidden_size)
        
        # Flatten embeddings
        flattened = embedded.reshape(batch_size, -1)  # (batch_size, context_length * hidden_size)
        
        # Hidden layer
        hidden = self.relu(flattened @ self.W_hidden + self.b_hidden)
        
        # Output layer
        logits = hidden @ self.W_output + self.b_output
        
        return logits, hidden, embedded
    
    def predict_proba(self, x):
        """Get probability distribution over next words"""
        logits, _, _ = self.forward(x)
        return self.softmax(logits)


class WordPredictor:
    """Main class for training and using the next-word prediction model"""
    
    def __init__(self, context_length: int = 5, vocab_size: int = 5000):
        self.context_length = context_length
        self.max_vocab_size = vocab_size
        self.word2idx = {}
        self.idx2word = {}
        self.model = None
        
    def build_vocabulary(self, text: str) -> None:
        """Build vocabulary from text"""
        words = text.lower().split()
        
        # Count word frequencies
        word_counts = Counter(words)
        
        # Take most common words
        most_common = word_counts.most_common(self.max_vocab_size - 2)
        
        # Build vocabulary with special tokens
        self.word2idx = {'<UNK>': 0, '<PAD>': 1}
        self.idx2word = {0: '<UNK>', 1: '<PAD>'}
        
        for idx, (word, _) in enumerate(most_common, start=2):
            self.word2idx[word] = idx
            self.idx2word[idx] = word
            
        print(f"Vocabulary built: {len(self.word2idx)} words")
        
    def prepare_training_data(self, text: str) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare training sequences from text"""
        words = text.lower().split()
        sequences = []
        targets = []
        
        # Create sequences
        for i in range(len(words) - self.context_length):
            context = words[i:i + self.context_length]
            target = words[i + self.context_length]
            
            # Convert to indices
            context_indices = [self.word2idx.get(w, 0) for w in context]
            target_idx = self.word2idx.get(target, 0)
            
            sequences.append(context_indices)
            targets.append(target_idx)
            
        X = np.array(sequences, dtype=np.int32)
        y = np.array(targets, dtype=np.int32)
        
        print(f"Created {len(sequences)} training examples")
        return X, y
    
    def train(self, text: str, epochs: int = 50, learning_rate: float = 0.01,
              batch_size: int = 32, time_limit_minutes: float = None) -> None:
        """Train the model on provided text"""
        print("Building vocabulary...")
        self.build_vocabulary(text)
        
        print("Preparing training data...")
        X, y = self.prepare_training_data(text)
        
        # Initialize model
        vocab_size = len(self.word2idx)
        self.model = SimpleNextWordModel(vocab_size, self.context_length, hidden_size=128)
        
        print(f"\nStarting training for {epochs} epochs...")
        print(f"Training on {len(X)} examples")
        
        start_time = time.time()
        n_batches = len(X) // batch_size
        
        for epoch in range(epochs):
            # Check time limit
            if time_limit_minutes and (time.time() - start_time) / 60 > time_limit_minutes:
                print(f"\nReached time limit of {time_limit_minutes} minutes")
                break
            
            # Shuffle data
            indices = np.random.permutation(len(X))
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            epoch_loss = 0
            
            # Mini-batch training
            for batch in range(n_batches):
                batch_start = batch * batch_size
                batch_end = batch_start + batch_size
                
                X_batch = X_shuffled[batch_start:batch_end]
                y_batch = y_shuffled[batch_start:batch_end]
                
                # Forward pass
                logits, hidden, embedded = self.model.forward(X_batch)
                probs = self.model.softmax(logits)
                
                # Compute loss (cross-entropy)
                batch_loss = -np.mean(np.log(probs[np.arange(batch_size), y_batch] + 1e-10))
                epoch_loss += batch_loss
                
                # Backward pass (simplified gradient descent)
                # Gradient of loss w.r.t. logits
                d_logits = probs.copy()
                d_logits[np.arange(batch_size), y_batch] -= 1
                d_logits /= batch_size
                
                # Gradient of output layer
                d_W_output = hidden.T @ d_logits
                d_b_output = np.sum(d_logits, axis=0)
                d_hidden = d_logits @ self.model.W_output.T
                
                # Gradient through ReLU
                d_hidden[hidden <= 0] = 0
                
                # Gradient of hidden layer
                flattened = embedded.reshape(batch_size, -1)
                d_W_hidden = flattened.T @ d_hidden
                d_b_hidden = np.sum(d_hidden, axis=0)
                d_flattened = d_hidden @ self.model.W_hidden.T
                
                # Gradient of embeddings
                d_embedded = d_flattened.reshape(batch_size, self.context_length, self.model.hidden_size)
                
                # Update embeddings
                for i in range(batch_size):
                    for j in range(self.context_length):
                        idx = X_batch[i, j]
                        self.model.W_embed[idx] -= learning_rate * d_embedded[i, j]
                
                # Update weights
                self.model.W_output -= learning_rate * d_W_output
                self.model.b_output -= learning_rate * d_b_output
                self.model.W_hidden -= learning_rate * d_W_hidden
                self.model.b_hidden -= learning_rate * d_b_hidden
            
            # Print progress
            avg_loss = epoch_loss / n_batches
            if (epoch + 1) % 5 == 0:
                elapsed = (time.time() - start_time) / 60
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}, Time: {elapsed:.1f}min")
        
        print("\nTraining complete!")
    
    def predict_next_words(self, context: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """Predict the next word given a context string"""
        if self.model is None:
            raise ValueError("Model not trained yet!")
        
        # Prepare context
        words = context.lower().split()
        
        # Take last context_length words
        if len(words) > self.context_length:
            words = words[-self.context_length:]
        elif len(words) < self.context_length:
            # Pad if necessary
            words = ['<PAD>'] * (self.context_length - len(words)) + words
        
        # Convert to indices
        indices = [self.word2idx.get(w, 0) for w in words]
        x = np.array([indices], dtype=np.int32)
        
        # Predict
        probs = self.model.predict_proba(x)[0]
        
        # Get top k predictions
        top_indices = np.argsort(probs)[-top_k:][::-1]
        
        predictions = []
        for idx in top_indices:
            word = self.idx2word[idx]
            prob = probs[idx]
            if word not in ['<UNK>', '<PAD>']:
                predictions.append((word, prob))
        
        return predictions
    
    def save(self, filepath: str) -> None:
        """Save model and vocabulary"""
        save_dict = {
            'model': self.model,
            'word2idx': self.word2idx,
            'idx2word': self.idx2word,
            'context_length': self.context_length,
            'max_vocab_size': self.max_vocab_size
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(save_dict, f)
        print(f"Model saved to {filepath}")
    
    def load(self, filepath: str) -> None:
        """Load model and vocabulary"""
        with open(filepath, 'rb') as f:
            save_dict = pickle.load(f)
        
        self.model = save_dict['model']
        self.word2idx = save_dict['word2idx']
        self.idx2word = save_dict['idx2word']
        self.context_length = save_dict['context_length']
        self.max_vocab_size = save_dict['max_vocab_size']
        
        print(f"Model loaded from {filepath}")


def main():
    """Interactive CLI for the word predictor"""
    predictor = WordPredictor(context_length=5, vocab_size=5000)
    
    print("=" * 60)
    print("Next-Word Prediction Language Model")
    print("Educational Tool for Understanding Language Models")
    print("=" * 60)
    print()
    
    while True:
        print("\nOptions:")
        print("1. Train on text file")
        print("2. Train on sample text")
        print("3. Predict next word")
        print("4. Save model")
        print("5. Load model")
        print("6. Exit")
        
        choice = input("\nEnter choice (1-6): ").strip()
        
        if choice == '1':
            filepath = input("Enter path to text file: ").strip()
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    text = f.read()
                
                epochs = int(input("Number of epochs (default 50): ") or "50")
                time_limit = input("Time limit in minutes (press Enter for no limit): ").strip()
                time_limit = float(time_limit) if time_limit else None
                
                predictor.train(text, epochs=epochs, time_limit_minutes=time_limit)
                
            except FileNotFoundError:
                print(f"File not found: {filepath}")
            except Exception as e:
                print(f"Error: {e}")
        
        elif choice == '2':
            print("\nUsing sample text (Shakespeare-style)...")
            sample_text = """
            to be or not to be that is the question whether tis nobler in the mind to suffer
            the slings and arrows of outrageous fortune or to take arms against a sea of troubles
            and by opposing end them to die to sleep no more and by a sleep to say we end
            the heart ache and the thousand natural shocks that flesh is heir to tis a consummation
            devoutly to be wished to die to sleep to sleep perchance to dream ay there is the rub
            for in that sleep of death what dreams may come when we have shuffled off this mortal coil
            must give us pause there is the respect that makes calamity of so long life
            """ * 20
            
            epochs = int(input("Number of epochs (default 50): ") or "50")
            predictor.train(sample_text, epochs=epochs)
        
        elif choice == '3':
            if predictor.model is None:
                print("Please train or load a model first!")
                continue
            
            context = input("\nEnter context (few words): ").strip()
            if context:
                predictions = predictor.predict_next_words(context, top_k=5)
                print(f"\nGiven: '{context}'")
                print("\nTop predictions:")
                for i, (word, prob) in enumerate(predictions, 1):
                    print(f"{i}. {word:15s} ({prob:.1%})")
        
        elif choice == '4':
            filepath = input("Enter save path (default: model.pkl): ").strip() or "model.pkl"
            predictor.save(filepath)
        
        elif choice == '5':
            filepath = input("Enter model path: ").strip()
            try:
                predictor.load(filepath)
            except FileNotFoundError:
                print(f"File not found: {filepath}")
            except Exception as e:
                print(f"Error: {e}")
        
        elif choice == '6':
            print("Goodbye!")
            break
        
        else:
            print("Invalid choice. Please try again.")


if __name__ == "__main__":
    main()
