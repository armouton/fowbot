# Student Guide: Exploring Language Models with FoWBot

## Introduction
This guide walks you through hands-on experiments with a small language model. You'll train models on different texts, adjust their settings, and observe how they learn — building intuition about the AI systems reshaping the world of work.

## Getting Started

### Setup
1. Download the FoWBot zip for your platform (Windows or Mac) from the link provided by your instructor
2. Extract the zip file
3. **Windows:** Double-click `FowBot.exe`
4. **Mac:** Double-click `FowBot`, then right-click > Open if you see a security warning (first time only)
5. The app opens in your browser automatically

### Your First Model (5 minutes)
1. Select **Shakespeare Sonnets (Medium)** from the dataset dropdown
2. Leave all other settings at their defaults
3. Click **Train** and watch the loss curve decrease
4. Once training finishes, type **to be or** in the prediction box and click **Predict**

You just trained a neural network! The loss curve shows the model's error decreasing as it learned patterns from Shakespeare's text.

## Core Exercises

### Exercise 1: Training Data Shapes Behavior (15 minutes)

**Goal:** See how different training data produces different "personalities."

1. Train on **Shakespeare Sonnets** (50 epochs, default settings)
2. Test these phrases and note the top predictions:
   - "to be or"
   - "the world is"
   - "love is"
3. Now train on **Dr. Seuss** (50 epochs, default settings)
4. Test with the **same phrases**

**Think about:**
- How do predictions differ between Shakespeare and Dr. Seuss?
- Which model gives more confident predictions (higher percentages)? Why?
- What does this tell you about where AI systems get their "personality"?

### Exercise 2: Training Duration and Diminishing Returns (15 minutes)

**Goal:** Observe how more training affects learning — and when it stops helping.

1. Train on **Shakespeare Sonnets** with **10 epochs**. Note the final loss.
2. Test with: "to be or"
3. Train again with **100 epochs**. Compare the loss curve shape.
4. Test with the same phrase.
5. Train again with **300 epochs**. Compare.

**Think about:**
- Is 10x more training 10x better?
- At what point does the loss curve flatten out?
- After 300 epochs, does the model just memorize the text rather than learning general patterns? (This is called *overfitting*.)

### Exercise 3: Learning Rate Instability (10 minutes)

**Goal:** See what happens when a model tries to learn too fast.

1. Train on **Shakespeare Sonnets** (100 epochs) with Learning Rate set to **Slow (0.5)**. Watch the loss curve — it should decrease smoothly.
2. Train again with Learning Rate set to **Fast (3.0)**. Watch closely.

**Think about:**
- What happens to the loss curve at the fast learning rate?
- Does the model still converge, or does it oscillate?
- What's the trade-off between learning speed and stability?

### Exercise 4: Simple vs. Advanced Architecture (15 minutes)

**Goal:** Compare the two model architectures.

1. Train on **Shakespeare Sonnets** (100 epochs) with Architecture set to **Simple (Feedforward)**
2. Set Words to **20** and generate from: "shall i compare thee"
3. Now train with Architecture set to **Advanced (LSTM)** (same settings)
4. Generate from the same prompt: "shall i compare thee"

**Think about:**
- Which architecture produces more coherent text?
- The Simple model sees all context words at once. The Advanced model reads them in sequence and decides what to remember. How does this difference show up in the output?
- Real systems like ChatGPT use an even more advanced architecture (transformers). What might that add?

### Exercise 5: Bias in Action (20 minutes)

**Goal:** Observe how training data bias appears in model outputs.

Train on each of the three included datasets and test all of them with the same phrases:
- "the world"
- "will be"
- "we must"

**Critical thinking:**
- How do predictions differ across datasets?
- Each dataset has its own "worldview" — can you identify it?
- Real AI systems are trained on internet text. Whose perspectives are overrepresented? Whose are missing?
- How might this affect AI used in hiring, lending, or healthcare?

### Exercise 6: Upload Your Own Text (10 minutes)

**Goal:** See the model learn from text you choose.

1. Find a text you're interested in — a news article, song lyrics, a speech, a chapter from a book. Save it as a `.txt` or `.docx` file.
2. Click **Choose file** in FoWBot and upload it
3. Train on your uploaded text (100 epochs)
4. Try generating text from a prompt that matches your source material

**Think about:**
- Does the model capture the style of your text?
- How much text do you need for the model to learn meaningful patterns?

## Reflection Questions

After completing the exercises, consider:

1. **Data is destiny:** What determines what a language model "knows"? What happens when training data is incomplete or biased?

2. **Scale matters:** Your model has thousands of parameters. GPT-4 has over a trillion. What becomes possible — and what becomes risky — at that scale?

3. **Architecture matters:** You saw that the LSTM outperformed the feedforward model. Transformers outperform LSTMs. How much of AI progress is about better architectures vs. more data vs. more compute?

4. **Cost of intelligence:** Your model trained in minutes on a laptop. Frontier models take months on supercomputers costing hundreds of millions of dollars. Who gets to build these systems? Who doesn't?

5. **Work implications:** Where would next-word prediction be useful in the workplace? Where would it be harmful? Which jobs are most affected?

## Key Takeaways

- Language models learn statistical patterns from training data — they don't "understand" language
- Training data determines model behavior, including biases
- More data and training generally help, but with diminishing returns and overfitting risk
- Architecture choices matter — sequential models capture patterns that simpler models miss
- Real AI systems face the same fundamental challenges at much larger scale
- Technical choices have social and ethical implications

## Troubleshooting

**App won't start:** Make sure you extracted the zip file first (don't run it from inside the zip).

**Browser doesn't open:** Go to http://localhost:5001 manually.

**Training seems stuck:** It's working if loss values are updating. LSTM and larger datasets are slower.

**Predictions seem random:** Train for more epochs (100+), or try a larger model size.

---

Created by Andre Mouton. Built with assistance from Claude (Anthropic).
