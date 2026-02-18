# Next-Word Prediction Language Model
## Educational Tool for Understanding AI and Language Models

### Overview
This is a simple, educational language model that predicts the next word in a sentence. It runs in your web browser and lets you:
- Train models on different datasets and watch the learning process in real time
- Upload your own text files (.txt or .docx) to train on
- Adjust model size, context window, learning rate, and training duration
- Generate multi-word sequences and see step-by-step predictions
- Compare how different training data shapes model behavior

### Requirements
- Python 3.8+
- No GPU needed - runs on any laptop CPU

### Quick Start

**Windows:**
1. Double-click `launch.bat`
2. Press any key when prompted to install dependencies
3. The app opens in your browser at http://localhost:5001

**Mac:**
1. Double-click `launch.command`
   - First time: you may need to right-click > Open to bypass Gatekeeper
2. Press any key when prompted to install dependencies
3. The app opens in your browser at http://localhost:5001

**Linux:**
1. Run `bash launch.sh` in a terminal
2. Press any key when prompted to install dependencies
3. The app opens in your browser at http://localhost:5001

The launcher finds your Python installation, installs numpy and flask (if needed), starts the server, and opens your browser.

### Using the App

**Training:**
1. **Choose a dataset** from the dropdown, or **upload your own** (.txt or .docx)
2. **Set parameters:**
   - **Epochs** (default 100) - how many training passes
   - **Learning Rate** - how fast the model learns (Slow/Medium/Fast)
   - **Model Size** - number of hidden units (Small 64 / Medium 256 / Large 512)
   - **Context Words** - how many previous words to consider (3 / 5 / 8)
3. **Click Train** and watch the loss curve decrease in real time

**Predicting:**
1. **Type a few words** in the prediction box
2. **Set Words** to 1 for next-word prediction, or higher for sequence generation
3. **Click Predict** to see the model's top predictions with confidence levels

### How It Works

**Model Architecture:**
- Feedforward neural network with word embeddings
- Configurable context window (3-8 previous words)
- Configurable hidden layer size (64-512 units)
- Vocabulary limited to 5,000 most common words

**Training Process:**
1. Builds vocabulary from your text
2. Creates training examples (sequences of N words -> next word)
3. Trains the neural network using gradient descent
4. Loss should decrease over time (the model is learning!)

**Parameter Effects:**
| Parameter | Larger = | Trade-off |
|-----------|----------|-----------|
| Epochs | Better learning | Longer training time |
| Learning Rate | Faster convergence | May overshoot (unstable) |
| Model Size | More capacity | Slower per epoch |
| Context Words | More context awareness | Needs more data to learn |

### Included Datasets

| File | Description | Size |
|------|-------------|------|
| `shakespeare.txt` | Shakespeare's Sonnets | ~2,600 lines |
| `drseuss.txt` | Dr. Seuss text | ~1,200 lines |

You can also upload your own `.txt` or `.docx` files through the web interface.

### Educational Experiments

**Experiment 1: Data Dependency**
- Train on Shakespeare, then predict "to be or"
- Train on Dr. Seuss, then predict "i do not"
- **Learning:** Models reflect their training data

**Experiment 2: Training Time**
- Try 10 epochs vs 100 epochs vs 300 epochs
- Watch how the loss curve changes
- **Learning:** More training helps, but with diminishing returns

**Experiment 3: Model Size**
- Train Small (64) vs Medium (256) vs Large (512) on the same data
- Compare prediction quality and training speed
- **Learning:** Larger models can learn more, but need more data and time

**Experiment 4: Context Window**
- Train with context 3 vs 5 vs 8
- Try predictions with short and long phrases
- **Learning:** More context helps the model disambiguate

**Experiment 5: Sequence Generation**
- Train on Shakespeare (100+ epochs), then generate 10 words from "shall i compare thee"
- **Learning:** Autoregressive generation shows how language models build text word by word

### Limitations (Important for Learning!)

1. **Only predicts words it has seen:** Unknown words are ignored
2. **Limited context:** Looks at the last 3-8 words only
3. **No understanding:** Just pattern matching, not comprehension
4. **Data bias:** Will reflect biases in training text
5. **Simplistic:** Real models (GPT, Claude) are much more sophisticated

### Files

```
FowBot/
  launch.bat              - Windows launcher (double-click)
  launch.command          - Mac launcher (double-click)
  launch.sh               - Linux launcher (bash launch.sh)
  app.py                  - Web server
  model.py                - Neural network model
  requirements.txt        - Python dependencies (numpy, flask)
  templates/index.html    - Web interface
  static/                 - CSS and JavaScript
  datasets/               - Training text files
```

### Common Issues

**App won't start**
- Make sure Python 3.8+ is installed
- On Windows: reinstall Python and check "Add Python to PATH"
- On Mac: try `brew install python3` or install from python.org

**Mac security warning ("unidentified developer")**
- Right-click `launch.command` > Open (first time only)
- Or: System Settings > Privacy & Security > click "Open Anyway"

**Browser doesn't open**
- Manually go to http://localhost:5001

**Training takes too long**
- Reduce epochs or use a smaller model size
- Use a smaller dataset

**Predictions seem random**
- Train for more epochs (100+ recommended)
- Use a larger dataset
- Try a larger model size

---

**Questions or issues?** This is a teaching tool - imperfections are features, not bugs!
