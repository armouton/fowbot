# Instructor Notes: Next-Word Prediction Language Model

## Overview
This educational package provides a browser-based language model that students can train on their own computers. It demonstrates core concepts of how language models work without requiring expensive GPU resources or technical expertise.

## What's Included

### Core Files
- **app.py** - Flask web server (serves the browser interface)
- **model.py** - Neural network implementation (NumPy only, ~170 lines)
- **launch.bat / launch.sh** - One-click launchers for Windows / Mac
- **requirements.txt** - Dependencies: numpy, flask

### Documentation
- **README.md** - Setup and usage instructions
- **STUDENT_GUIDE.md** - Structured exercises and learning activities
- **INSTRUCTOR_NOTES.md** - This file

### Datasets (in `datasets/` folder)
- **shakespeare.txt** - Shakespeare's Sonnets (~2,600 lines, best for training)
- **drseuss.txt** - Dr. Seuss text (~1,200 lines)
- **sample_large_dataset.txt** - Future of work articles (~1,000 words)
- **sample_future_of_work.txt** - Short future of work text (~330 words)

Students can add their own `.txt` files to the `datasets/` folder.

## Technical Specifications

### Model Architecture
- Feedforward neural network with word embeddings
- Input: Last 5 words (configurable in code)
- Hidden layer: 128 units with ReLU activation
- Output: Probability distribution over vocabulary
- Total parameters: ~500K-1M depending on vocabulary size

### Training
- Mini-batch gradient descent (batch size: 32)
- Loss: Cross-entropy
- Optimizer: Vanilla gradient descent with fixed learning rate
- Training time: 1-10 minutes depending on dataset and epochs

### Web Interface
- Flask backend on localhost:5001
- Real-time loss curve visualization (Chart.js)
- No internet required after initial setup (Chart.js bundled locally)
- Works on Windows and Mac

### Performance Expectations
This is intentionally a **simple** model. Students should expect:
- Modest prediction quality (not comparable to GPT)
- Clear learning curves (loss decreases visibly)
- Visible impact of dataset differences
- Some nonsensical predictions (teaching moment!)

## Setup for Distribution

### Preparing the Zip File
1. Make sure all files are present (run the app yourself first to verify)
2. Delete the `venv/` folder if one exists (students will create their own)
3. Delete any `__pycache__/` folders
4. Zip the entire FowBot folder
5. Upload to your LMS

### Student Prerequisites
- Python 3.8+ installed
- The launcher scripts handle virtual environment creation and dependency installation automatically
- Students on Windows should ensure "Add Python to PATH" was checked during Python installation

## Pedagogical Design

### Learning Objectives
Students will understand:
1. How language models learn from data
2. The relationship between training data and model behavior
3. Computational constraints of AI systems
4. Data bias and its manifestation in AI
5. The gap between simple educational models and production systems

### Why These Design Choices?

**Browser-based interface:**
- No command-line experience needed
- Visual feedback (loss curve) makes training intuitive
- Students can focus on concepts, not tooling

**Word-level (not character-level):**
- Predictions are readable words, not letter soup
- More intuitive for non-technical students

**Next-word (not full text generation):**
- Simpler to understand
- Clear evaluation: are the predictions sensible?
- Avoids complexity of sampling strategies

## Class Integration

### Suggested Timeline

**Before class:** Students download, unzip, and run `launch.bat`/`launch.sh` to verify setup works.

**In class (60-90 minutes):**
1. Brief demo of the tool (5 min)
2. Students complete Exercises 1-3 from STUDENT_GUIDE.md (30 min)
3. Small group discussion: compare observations (10 min)
4. Exercise 5: bias detection (15 min)
5. Full-class discussion on implications (15 min)

**After class:** Reflection paper connecting observations to course themes.

### Assessment Options

**Low-stakes:**
- Completion of exercises (participation credit)
- In-class observations shared with group

**Medium-stakes:**
- Comparative analysis: your simple model vs. ChatGPT
- Dataset bias analysis

**High-stakes:**
- Research paper connecting AI training to labor market implications
- Design policy recommendations for AI in workplace

## Common Student Questions

**Q: Why are the predictions so bad?**
A: This teaches realistic expectations. Real models have billions of parameters and train for weeks. The gap between this and ChatGPT is the point.

**Q: Can I make it better?**
A: Yes! More epochs, larger datasets. But discuss diminishing returns and computational limits.

**Q: How is this different from ChatGPT?**
A: Scale (millions vs. billions of parameters), data (kilobytes vs. terabytes), architecture (simple feedforward vs. transformer), resources (laptop vs. supercomputer).

## Troubleshooting

### Common Issues

**"Python not found" on Windows**
- Students need to reinstall Python from python.org and check "Add Python to PATH"

**Port 5001 already in use**
- Another instance may be running. Close the previous terminal window.
- Or edit `app.py` line at the bottom to change the port number.

**Launcher script won't run on Mac**
- May need: `chmod +x launch.sh` then `bash launch.sh`

**Training is very fast (finishes instantly)**
- This happens with very small datasets (sample_future_of_work.txt)
- Use shakespeare.txt or drseuss.txt for a more visible training process
- Increase epochs to 100-200

## Modifications

### Adding Datasets
Simply place `.txt` files in the `datasets/` folder. They appear automatically in the dropdown.

### Changing Model Parameters
Edit `app.py` line where `WordPredictor` is instantiated:
- `context_length=5` - number of previous words to consider
- `vocab_size=5000` - maximum vocabulary size

### Changing Port
Edit the last line of `app.py`: `app.run(port=5001)` - change 5001 to any available port.

---

**Version:** 2.0 (Web interface)
**Last Updated:** 2026
