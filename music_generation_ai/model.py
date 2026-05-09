import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM

def create_network(n_vocab):
    """
    Create the structure of the neural network.
    
    Args:
        n_vocab (int): The number of unique notes/chords in the vocabulary.
        
    Returns:
        model: A compiled Keras sequential model.
    """
    print("Building LSTM model architecture...")
    
    model = Sequential()
    
    # LSTM layer (512 units, return_sequences=True)
    # input_shape: Sequence length of 100, 1 feature (normalized note integer)
    model.add(LSTM(
        512,
        input_shape=(100, 1),
        return_sequences=True
    ))
    
    # Dropout layer (0.3)
    model.add(Dropout(0.3))
    
    # LSTM layer (512 units, return_sequences=False)
    model.add(LSTM(512, return_sequences=False))
    
    # Dropout layer (0.3)
    model.add(Dropout(0.3))
    
    # Dense layer (256 units, relu activation)
    model.add(Dense(256, activation='relu'))
    
    # Dropout layer (0.3)
    model.add(Dropout(0.3))
    
    # Dense output layer (n_vocab units, softmax activation)
    model.add(Dense(n_vocab, activation='softmax'))
    
    # Compile with: loss=categorical_crossentropy, optimizer=adam
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    
    # Print the model summary
    model.summary()
    
    return model

if __name__ == '__main__':
    # Test script to verify model creation
    print("Testing model build with a dummy vocab size of 100...")
    model = create_network(100)
    print("Model successfully built.")
