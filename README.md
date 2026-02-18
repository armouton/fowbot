# FoWBot: A Tiny Trainable Language Model
## Educational Tool for ECN373 — The Future of Work

### Overview
FoWBot is a browser-based tool that lets you train and explore small neural-network language models. You can:
- Train feedforward or LSTM models on different texts and watch the learning process in real time
- Upload your own text files (.txt or .docx)
- Adjust architecture, model size, context window, learning rate, and training duration
- Generate multi-word sequences and see step-by-step predictions
- Compare how different settings and architectures shape model behavior

### Getting Started

**Download the app** from the [Releases page](../../releases):
- **Windows:** Download `FowBot-Windows.zip`, extract, and double-click `FowBot.exe`
- **Mac:** Download `FowBot-Mac.zip`, extract, and double-click `FowBot`
  - First time: right-click > Open to bypass Gatekeeper

The app starts a local server and opens in your browser. No Python installation or setup required.

### Running from Source (alternative)

If you prefer to run from source (requires Python 3.8+):

**Windows:** Double-click `launch.bat`
**Mac:** Double-click `launch.command` (right-click > Open first time)
**Linux:** Run `bash launch.sh`

The launcher installs dependencies (numpy, flask) and starts the server at http://localhost:5001.

### Using the App

**Training:**
1. Choose a dataset from the dropdown, or upload your own (.txt or .docx)
2. Set parameters:
   - **Epochs** (default 100) — how many training passes
   - **Learning Rate** — Slow (0.5) / Medium (1.0) / Fast (3.0)
   - **Architecture** — Simple (Feedforward) or Advanced (LSTM)
   - **Model Size** — Small (64) / Medium (256) / Large (512) hidden units
   - **Context Words / Sequence Length** — 3 / 5 / 10 / 20
3. Click **Train** and watch the loss curve decrease in real time

**Predicting:**
1. Type a few words in the prediction box
2. Set **Words** to 1 for next-word prediction, or higher for sequence generation (up to 500)
3. Click **Predict** to see the model's top predictions with confidence levels

### Architectures

**Simple (Feedforward):** Concatenates embeddings of the previous N words and passes them through a hidden layer. Fast to train, but treats context as a fixed window with no notion of word order beyond position.

**Advanced (LSTM):** Processes context words sequentially through gates that control what to remember and forget. Slower to train, but can capture sequential patterns and produce more coherent generated text.

Both architectures are implemented in pure NumPy — no deep learning frameworks required.

### Included Datasets

| File | Description |
|------|-------------|
| Dr. Seuss (Small) | Short, repetitive text — good for quick experiments |
| Shakespeare Sonnets (Medium) | Poetic English with rich vocabulary |
| Shakespeare Plays (Large) | Longer text for more thorough training |

You can also upload your own `.txt` or `.docx` files through the web interface.

### Educational Experiments

**Data dependency:** Train on Shakespeare, then predict "to be or". Train on Dr. Seuss, then predict "i do not". Models reflect their training data.

**Training time:** Try 10 vs 100 vs 300 epochs. Watch the loss curve — more training helps, but with diminishing returns and eventual overfitting.

**Learning rate instability:** Train with Fast (3.0) learning rate and observe oscillations in the loss curve. Compare with Slow (0.5) for smooth convergence.

**Architecture comparison:** Train both Simple and Advanced on the same text, then generate 20 words from the same prompt. Compare coherence.

**Overfitting:** Train for 200+ epochs on a small dataset. The loss approaches zero, but generated text starts repeating training data verbatim rather than generalizing.

### Files

```
FowBot/
  app.py                  - Web server
  model.py                - Neural network models (feedforward + LSTM)
  requirements.txt        - Python dependencies (numpy, flask)
  fowbot.spec             - PyInstaller build configuration
  templates/index.html    - Web interface
  static/                 - CSS and JavaScript
  datasets/               - Training text files
  launch.bat              - Windows launcher (run from source)
  launch.command          - Mac launcher (run from source)
  launch.sh               - Linux launcher (run from source)
  .github/workflows/      - CI build for Windows and Mac executables
```

### Common Issues

**Mac security warning ("unidentified developer")**
Right-click the app > Open (first time only), or: System Settings > Privacy & Security > Open Anyway.

**Browser doesn't open**
Manually go to http://localhost:5001

**Predictions seem random**
Train for more epochs (100+ recommended) or use a larger model size.

---

Created by Andre Mouton. Built with assistance from Claude (Anthropic).
