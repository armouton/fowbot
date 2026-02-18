# Student Guide: Exploring Language Models

## Introduction
This guide will help you learn about language models through hands-on experimentation. You'll train models on different datasets using a web-based tool and observe how they behave.

## Getting Started

### Setup (2 minutes)
1. Download and unzip the FowBot folder
2. **Windows:** Double-click `launch.bat`
3. **Mac:** Open Terminal in the folder and run `bash launch.sh`
4. The app opens in your browser at http://localhost:5001

### First Model (5 minutes)
Train your first model to get familiar:
- Select **"sample large dataset"** from the dropdown
- Set epochs to **30**
- Click **Train** and watch the loss curve
- Once done, type **"the future of"** in the prediction box and click **Predict**

**What you're observing:** The model learned patterns from a text about the future of work. The loss curve shows the model's error decreasing as it learns.

## Core Exercises

### Exercise 1: The Impact of Training Data (15 minutes)

**Objective:** Understand how training data shapes model behavior.

**Steps:**
1. Select **"shakespeare"** and train with 50 epochs. Watch the loss curve.
2. Test with these phrases:
   - "to be or"
   - "the world is"
   - "love is"
3. Now select **"drseuss"** and train with 50 epochs.
4. Test with the **same phrases** as above.

**Questions to consider:**
- How do the predictions differ between Shakespeare and Dr. Seuss?
- Which model gives more confident (higher percentage) predictions? Why?
- What does this tell you about where AI models get their "personality"?

### Exercise 2: Training Time vs. Quality (15 minutes)

**Objective:** See how training duration affects learning.

**Steps:**
1. Select **"shakespeare"** and train with **10 epochs**. Note the final loss value.
2. Test with: "to be or"
3. Train again with **50 epochs**. Compare the loss curve.
4. Test with the same phrase.
5. Train again with **200 epochs**. Compare again.

**Watch for:**
- How the shape of the loss curve changes
- Whether predictions improve with more training
- When the loss stops decreasing much (diminishing returns)

**Questions:**
- Is 5x more training 5x better?
- At what point does additional training stop helping?
- How would you balance training time vs. quality in a real system?

### Exercise 3: Dataset Size (15 minutes)

**Objective:** Observe how dataset size affects learning.

**Steps:**
1. Train on **"sample future of work"** (small, ~330 words) with 50 epochs
2. Test with: "the future of"
3. Train on **"sample large dataset"** (~1,000 words) with 50 epochs
4. Test with the same phrase
5. Train on **"shakespeare"** (~2,600 lines) with 50 epochs
6. Test with: "the world is"

**Observations to make:**
- Compare the loss curves - which dataset produces lower loss?
- Compare prediction confidence (the percentages)
- Does a larger dataset always mean better predictions?

### Exercise 4: Context Awareness (10 minutes)

**Objective:** See how context affects predictions.

Train on **"shakespeare"** (50 epochs), then test with:
- "to" (just 1 word)
- "to be" (2 words)
- "to be or" (3 words)
- "to be or not" (4 words)

**Questions:**
- Do predictions change as you add more context?
- Which gives more specific/confident predictions?
- What does this tell you about how language models work?

### Exercise 5: Bias in Action (20 minutes)

**Objective:** Observe how training data bias appears in predictions.

Train on each of the included datasets and test all of them with the same phrases:
- "the world"
- "will be"
- "we must"

**Critical thinking:**
- How do predictions differ across datasets?
- Each dataset has its own "worldview" - can you identify it?
- What are implications for real AI systems trained on internet text?
- How might this affect AI used in hiring, loans, or healthcare?

## Reflection Questions

After completing the exercises, consider:

1. **Data is destiny:** What determines what a language model "knows"?

2. **Scale matters:** Your model has ~100K parameters. GPT-4 has over a trillion. What becomes possible at that scale?

3. **Context windows:** Your model looks at 5 words. Modern LLMs can handle 100,000+ tokens. How does this change capabilities?

4. **Computational cost:** Your model trained in minutes. Large models take weeks on supercomputers. Who can afford to build frontier AI?

5. **Real-world applications:** Where would next-word prediction be useful? Where would it be harmful?

## Discussion Topics

### For Class Discussion:

**Ethics:**
- If models reflect their training data, whose perspectives are represented in major AI systems?
- What responsibility do AI companies have for model outputs?

**Work & Employment:**
- Which jobs are most vulnerable to automation by language models?
- What skills become more valuable as AI improves?
- How should society manage AI-driven job displacement?

**Trust & Verification:**
- Your model sometimes makes mistakes. How do you know when to trust AI?
- When do these errors matter? When are they just amusing?

## Key Takeaways

By the end of these exercises, you should understand:
- Language models learn statistical patterns from training data
- Training data determines model behavior and biases
- More data and training generally improve performance (with diminishing returns)
- Context helps models make better predictions
- Real-world AI systems face the same challenges at much larger scale
- Technical choices have social and ethical implications

## Troubleshooting

**App won't start**
- Make sure Python 3.8+ is installed
- On Windows, check that "Add Python to PATH" was selected during installation

**Browser doesn't open automatically**
- Go to http://localhost:5001 manually

**Training seems stuck**
- Training is working if the loss values are updating. Larger datasets take longer per epoch.
- You can click Stop and try with fewer epochs.

**Predictions seem random**
- Train for more epochs, or use a larger dataset

Remember: The goal isn't to build a perfect model, but to understand how these systems work and what their implications are for the future of work!
