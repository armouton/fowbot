# Instructor Notes: FoWBot

## Overview
FoWBot is a browser-based tool that lets students train and interact with small language models. It demonstrates core concepts of how language models learn from data, without requiring GPU resources or technical expertise. Students can compare two neural network architectures (feedforward and LSTM) and experiment with training parameters to build intuition about model behavior.

## Distribution

### Recommended: Pre-built executables
Download Windows and Mac builds from the [Releases page](../../releases). Students extract the zip and double-click to run — no Python installation needed.

- **Windows:** `FowBot-Windows.zip` → extract → double-click `FowBot.exe`
- **Mac:** `FowBot-Mac.zip` → extract → right-click > Open (first time only, due to Gatekeeper)

### Alternative: Run from source
Students with Python 3.8+ can use the launcher scripts (`launch.bat` / `launch.command`). This requires a working Python installation and is more error-prone.

## What's Included

### Core files
- **app.py** — Flask web server
- **model.py** — Feedforward and LSTM implementations (pure NumPy)
- **templates/index.html** — Web interface with live training visualization

### Datasets
| File | Size | Best for |
|------|------|----------|
| Dr. Seuss (Small) | ~1,200 lines | Quick experiments, visible overfitting |
| Shakespeare Sonnets (Medium) | ~2,600 lines | Core exercises |
| Shakespeare Plays (Large) | Larger text | Architecture comparison, longer training |

Students can also upload their own `.txt` or `.docx` files.

## Technical Specifications

### Architectures
- **Simple (Feedforward):** Concatenates word embeddings, single hidden layer, ReLU activation. Fast training, no sequential awareness.
- **Advanced (LSTM):** Sequential processing with forget/input/output gates. Slower training, better coherence in generated text.

Both are implemented in pure NumPy with no deep learning frameworks.

### Training parameters
| Parameter | Options | Notes |
|-----------|---------|-------|
| Epochs | 1–500 | 100 is a good default |
| Learning Rate | Slow (0.5) / Medium (1.0) / Fast (3.0) | Fast shows instructive instability |
| Model Size | Small (64) / Medium (256) / Large (512) | Hidden layer units |
| Context / Sequence Length | 3 / 5 / 10 / 20 | Input window size |
| Architecture | Simple (Feedforward) / Advanced (LSTM) | |

### Performance expectations
This is intentionally a simple model. Students should expect:
- Modest prediction quality (not comparable to ChatGPT)
- Clear learning curves (loss decreases visibly)
- Visible impact of parameter and architecture choices
- Overfitting on small datasets after 100+ epochs
- Some nonsensical predictions (this is a teaching moment)

## Pedagogical Design

### Learning objectives
Students will understand:
1. How language models learn statistical patterns from data
2. The relationship between training data and model behavior
3. How architecture choices (feedforward vs. LSTM) affect capabilities
4. Parameter trade-offs: learning rate instability, overfitting, model capacity
5. The gap between simple models and production systems (GPT, Claude)
6. Data bias and its manifestation in AI outputs

### Key teaching moments
- **Learning rate at Fast (3.0):** Loss curve oscillates visibly — demonstrates gradient descent instability
- **Overfitting:** After 200+ epochs on Dr. Seuss, the model memorizes rather than generalizes
- **Architecture comparison:** LSTM generates more coherent sequences than feedforward from the same training data
- **Data bias:** Training on different texts produces very different "worldviews" from the same prompts

### Why these design choices?
- **Browser-based:** No command-line experience needed; visual feedback makes training intuitive
- **Word-level predictions:** More intuitive than character-level for non-technical students
- **Two architectures:** Lets students see concretely how model structure matters, not just data and scale
- **Adjustable parameters:** Students can discover trade-offs through experimentation

## Class Integration

### Suggested timeline

**Before class:** Students download and run FoWBot to verify it works.

**In class (60–90 minutes):**
1. Brief demo of the tool (5 min)
2. Students complete exercises from STUDENT_GUIDE.md (40 min)
3. Small group discussion: compare observations (10 min)
4. Full-class discussion on implications for work and society (15 min)

**After class:** Reflection paper connecting observations to course themes.

### Assessment options

**Low-stakes:** Completion of exercises (participation credit), in-class observations shared with group.

**Medium-stakes:** Comparative analysis of architectures, or dataset bias analysis using uploaded texts.

**High-stakes:** Research paper connecting AI training dynamics to labor market implications.

## Common Student Questions

**Q: Why are the predictions so bad?**
A: This is the point. Real models have billions of parameters and train on terabytes of data. The gap between FoWBot and ChatGPT illustrates what scale provides.

**Q: What's the difference between the two architectures?**
A: The feedforward model sees all context words at once (like reading a sentence with all words jumbled). The LSTM reads them in order and decides what to remember — more like how you read.

**Q: Can I make it better?**
A: More epochs, larger model, more data. But discuss diminishing returns and computational limits.

## Troubleshooting

**Mac Gatekeeper warning:** Right-click > Open (first time only).

**Port 5001 already in use:** Close the previous FoWBot instance, or the previous terminal window.

**Training seems stuck:** Training is working if loss values are updating. Larger datasets and LSTM are slower per epoch.

---

Created by Andre Mouton. Built with assistance from Claude (Anthropic).
