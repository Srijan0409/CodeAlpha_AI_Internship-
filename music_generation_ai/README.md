# AI Music Generation System

An end-to-end deep learning system that learns musical patterns from MIDI files and generates original compositions using a Long Short-Term Memory (LSTM) neural network. This is a Code Alpha internship project.

## 🎵 Project Overview

This system processes classical or jazz music, learns the sequence of notes and chords, and uses deep learning to generate brand new sequences that are converted back into playable MIDI files.

**Key Features:**
1. **MIDI Parsing:** Extracts individual notes and chords from standard `.mid` files using `music21`.
2. **Deep Learning Model:** Utilizes a stacked LSTM architecture built with TensorFlow/Keras to learn sequential music patterns.
3. **Music Generation:** Predicts the next notes in a sequence recursively to build an entirely new 500-note composition.

## 📁 Folder Structure

```text
music_generation_ai/
├── midi_data/           ← (Place your training .mid files here)
├── preprocess.py        ← Parses MIDI files and saves extracted sequences
├── model.py             ← Contains the LSTM network architecture
├── train.py             ← Trains the network and saves the model
├── generate.py          ← Generates new music and creates output.mid
├── requirements.txt     ← Project dependencies
└── README.md            ← You are here
```

## 🛠️ Prerequisites

Make sure you have Python 3.8+ installed. You can install the required libraries using `pip`:

```bash
pip install -r requirements.txt
```

## 🚀 How to Run the Project Step-by-Step

### 1. Add MIDI Data
Before you can train the model, you need a dataset. Place a few classical or jazz `.mid` files inside the `midi_data/` folder. The more files you provide, the better the neural network will learn to generalize.

### 2. Preprocess the Data
Run the preprocessing script to extract the notes and chords from your MIDI files. This will save a flattened sequence of strings to a file named `notes.pkl`.

```bash
python preprocess.py
```

### 3. Verify the Architecture (Optional)
You can run the `model.py` file directly just to verify that the TensorFlow structure builds correctly without errors. It will print a model summary to the console.

```bash
python model.py
```

### 4. Train the Model
Run the training script. It will load `notes.pkl`, prepare sliding window sequences (100 notes each), and train the stacked LSTM model for 100 epochs. A checkpoint is saved every 10 epochs.

```bash
python train.py
```
*(Note: Training deep neural networks can be computationally expensive. Running this on a machine with a dedicated GPU is highly recommended.)*

### 5. Generate New Music
Once training is complete and `music_model.h5` is saved, run the generation script. It will pick a random starting sequence from the training data, predict 500 subsequent notes, and output them as a new MIDI file.

```bash
python generate.py
```

The final output will be saved as `output.mid` in the root folder. You can play this file using any standard media player, or import it into a Digital Audio Workstation (DAW) to assign custom instruments!

## 🧠 Neural Network Architecture

The model is built using Keras and consists of:
*   **LSTM Layer:** 512 units (returns sequences)
*   **Dropout Layer:** 30% to prevent overfitting
*   **LSTM Layer:** 512 units
*   **Dropout Layer:** 30%
*   **Dense Layer:** 256 units (ReLU activation)
*   **Dropout Layer:** 30%
*   **Output Dense Layer:** Vocab size (Softmax activation)

Loss function used is `categorical_crossentropy` and it is optimized using `adam`.
