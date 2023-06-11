from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.regularizers import l2

def generate_dot_file():
    model = Sequential([
        Dense(128, input_shape=(X_train.shape[1],), activation='sigmoid'),
        Dense(64, activation='tanh'),
        Dense(32, activation='relu', kernel_regularizer=l2(0.01)),
        Dense(16, activation='relu'),
        Dense(1, activation='linear')
    ])
    
    dot_content = 'digraph Model {\n'
    
    # Input layer
    dot_content += '    input [shape=box, label="Input"]\n'
    
    # Hidden layers
    for i, layer in enumerate(model.layers):
        layer_name = f'hidden_{i}'
        layer_label = f'{layer.name}\n({layer.output_shape[1:]})'
        dot_content += f'    {layer_name} [shape=circle, label="{layer_label}"]\n'
    
    # Output layer
    dot_content += f'    output [shape=box, label="Output"]\n'
    
    # Connect layers
    dot_content += '    input -> hidden_0\n'
    for i in range(len(model.layers) - 1):
        dot_content += f'    hidden_{i} -> hidden_{i+1}\n'
    dot_content += f'    hidden_{len(model.layers) - 1} -> output\n'
    
    dot_content += '}'
    
    with open('model.dot', 'w') as dot_file:
        dot_file.write(dot_content)

generate_dot_file()
