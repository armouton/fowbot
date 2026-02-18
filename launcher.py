#!/usr/bin/env python3
"""
Simple launcher for the Next-Word Predictor
Provides quick training options for students
"""

import subprocess
import sys
import os

def check_numpy():
    """Check if numpy is installed"""
    try:
        import numpy
        return True
    except ImportError:
        return False

def main():
    print("=" * 60)
    print("Next-Word Predictor Launcher")
    print("=" * 60)
    print()
    
    # Check dependencies
    if not check_numpy():
        print("⚠️  NumPy is not installed!")
        print("Please install it first: pip install numpy")
        print()
        response = input("Try to install now? (y/n): ").strip().lower()
        if response == 'y':
            print("Installing numpy...")
            subprocess.run([sys.executable, "-m", "pip", "install", "numpy"])
            print()
        else:
            return
    
    print("Welcome! This tool helps you understand how language models learn.")
    print()
    print("Quick Start Options:")
    print("1. Try the demo (Shakespeare text, fast)")
    print("2. Train on 'Future of Work' dataset")
    print("3. Train on your own text file")
    print("4. Open the full interactive program")
    print("5. Exit")
    print()
    
    choice = input("Choose an option (1-5): ").strip()
    
    if choice == '1':
        print("\n🎭 Training on Shakespeare-style text with 30 epochs...")
        print("This will take about 1-2 minutes.\n")
        
        # Create input script
        script = """2
30
3
to be or
3
to sleep
3
that is the
6
"""
        
        # Run with automated input
        process = subprocess.Popen(
            [sys.executable, "next_word_predictor_numpy.py"],
            stdin=subprocess.PIPE,
            text=True
        )
        process.communicate(input=script)
        
    elif choice == '2':
        print("\n💼 Training on 'Future of Work' dataset...")
        print()
        epochs = input("How many epochs? (recommended: 50-100): ").strip() or "50"
        
        script = f"""1
sample_large_dataset.txt
{epochs}

3
the future of
3
workers need to
3
artificial intelligence
4
future_of_work_model.pkl
6
"""
        
        process = subprocess.Popen(
            [sys.executable, "next_word_predictor_numpy.py"],
            stdin=subprocess.PIPE,
            text=True
        )
        process.communicate(input=script)
        
        print("\n✓ Model saved as 'future_of_work_model.pkl'")
        print("You can load it later to make more predictions!")
        
    elif choice == '3':
        filepath = input("\nEnter the path to your text file: ").strip()
        
        if not os.path.exists(filepath):
            print(f"⚠️  File not found: {filepath}")
            return
        
        print()
        epochs = input("How many epochs? (recommended: 50): ").strip() or "50"
        
        script = f"""1
{filepath}
{epochs}

3
your test context here
6
"""
        
        print(f"\n📚 Training on {filepath}...")
        print("After training, you can test predictions.")
        print("Type your test phrases when prompted.\n")
        
        # Just launch the program and let user interact
        subprocess.run([sys.executable, "next_word_predictor_numpy.py"])
        
    elif choice == '4':
        print("\n🚀 Launching full program...\n")
        subprocess.run([sys.executable, "next_word_predictor_numpy.py"])
        
    elif choice == '5':
        print("\nGoodbye! 👋")
        return
        
    else:
        print("\n⚠️  Invalid choice. Please run again and choose 1-5.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted. Goodbye!")
    except Exception as e:
        print(f"\n⚠️  An error occurred: {e}")
        print("Please report this if the issue persists.")
