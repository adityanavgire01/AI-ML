import numpy as np
import matplotlib.pyplot as plt

# Set random seed for reproducible results
np.random.seed(42)

class XORNeuralNetwork:
    def __init__(self, input_size=2, hidden_size=4, output_size=1, learning_rate=0.5):
        """
        Initialize the neural network with random weights and biases
        
        Architecture: Input(2) -> Hidden(4) -> Output(1)
        """
        self.learning_rate = learning_rate
        
        # Initialize weights with small random values
        # Why small? Large weights can cause vanishing/exploding gradients
        self.W1 = np.random.randn(input_size, hidden_size) * 0.5  # 2x4 matrix
        self.b1 = np.zeros((1, hidden_size))                      # 1x4 bias vector
        self.W2 = np.random.randn(hidden_size, output_size) * 0.5 # 4x1 matrix
        self.b2 = np.zeros((1, output_size))                      # 1x1 bias
        
        # Store training history for visualization
        self.loss_history = []
        
    def sigmoid(self, x):
        """
        Sigmoid activation function: σ(x) = 1 / (1 + e^(-x))
        
        Why sigmoid?
        - Outputs values between 0 and 1 (good for probabilities)
        - Smooth and differentiable
        - Non-linear (essential for solving XOR)
        """
        # Clip x to prevent overflow
        x = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self, x):
        """
        Derivative of sigmoid: σ'(x) = σ(x) * (1 - σ(x))
        
        This is used in backpropagation to calculate gradients
        """
        s = self.sigmoid(x)
        return s * (1 - s)
    
    def forward_propagation(self, X):
        """
        Forward pass: compute predictions
        
        Flow: Input -> Hidden Layer -> Output Layer
        """
        # Hidden layer computation
        # z1 = X * W1 + b1 (linear transformation)
        self.z1 = np.dot(X, self.W1) + self.b1
        # a1 = σ(z1) (activation)
        self.a1 = self.sigmoid(self.z1)
        
        # Output layer computation
        # z2 = a1 * W2 + b2 (linear transformation)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        # a2 = σ(z2) (final prediction)
        self.a2 = self.sigmoid(self.z2)
        
        return self.a2
    
    def compute_loss(self, y_true, y_pred):
        """
        Binary Cross-Entropy Loss
        
        Loss = -[y*log(ŷ) + (1-y)*log(1-ŷ)]
        
        Why this loss?
        - Penalizes wrong predictions more heavily
        - Works well with sigmoid output
        """
        # Add small epsilon to prevent log(0)
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    
    def backward_propagation(self, X, y):
        """
        Backpropagation: compute gradients and update weights
        
        This is where the "learning" happens!
        """
        m = X.shape[0]  # number of training examples
        
        # Calculate error at output layer
        # How far off were our predictions?
        dz2 = self.a2 - y  # Error signal
        
        # Calculate gradients for output layer weights and biases
        dW2 = (1/m) * np.dot(self.a1.T, dz2)  # Gradient w.r.t. W2
        db2 = (1/m) * np.sum(dz2, axis=0, keepdims=True)  # Gradient w.r.t. b2
        
        # Propagate error back to hidden layer
        # Chain rule: how does hidden layer error affect output error?
        dz1 = np.dot(dz2, self.W2.T) * self.sigmoid_derivative(self.z1)
        
        # Calculate gradients for hidden layer weights and biases
        dW1 = (1/m) * np.dot(X.T, dz1)  # Gradient w.r.t. W1
        db1 = (1/m) * np.sum(dz1, axis=0, keepdims=True)  # Gradient w.r.t. b1
        
        # Update weights using gradient descent
        # New_weight = Old_weight - learning_rate * gradient
        self.W2 -= self.learning_rate * dW2
        self.b2 -= self.learning_rate * db2
        self.W1 -= self.learning_rate * dW1
        self.b1 -= self.learning_rate * db1
    
    def train(self, X, y, epochs=1000, verbose=True):
        """
        Train the neural network
        """
        for epoch in range(epochs):
            # Forward pass
            predictions = self.forward_propagation(X)
            
            # Calculate loss
            loss = self.compute_loss(y, predictions)
            self.loss_history.append(loss)
            
            # Backward pass
            self.backward_propagation(X, y)
            
            # Print progress
            if verbose and epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.6f}")
    
    def predict(self, X):
        """
        Make predictions (returns probabilities)
        """
        return self.forward_propagation(X)
    
    def predict_binary(self, X):
        """
        Make binary predictions (0 or 1)
        """
        predictions = self.predict(X)
        return (predictions > 0.5).astype(int)

# Create XOR dataset
print("=== XOR PROBLEM ===")
print("XOR Truth Table:")
print("Input1 | Input2 | Output")
print("   0   |   0    |   0")
print("   0   |   1    |   1") 
print("   1   |   0    |   1")
print("   1   |   1    |   0")
print()

# XOR data
X = np.array([[0, 0],
              [0, 1], 
              [1, 0],
              [1, 1]])

y = np.array([[0],
              [1],
              [1], 
              [0]])

print("Training Data:")
for i in range(len(X)):
    print(f"Input: {X[i]} -> Expected Output: {y[i][0]}")
print()

# Create and train the neural network
print("=== TRAINING NEURAL NETWORK ===")
nn = XORNeuralNetwork(input_size=2, hidden_size=4, output_size=1, learning_rate=0.5)

print("Initial Predictions (before training):")
initial_predictions = nn.predict(X)
for i in range(len(X)):
    print(f"Input: {X[i]} -> Prediction: {initial_predictions[i][0]:.4f}")
print()

# Train the network
nn.train(X, y, epochs=2000, verbose=True)

print("\n=== RESULTS AFTER TRAINING ===")
final_predictions = nn.predict(X)
binary_predictions = nn.predict_binary(X)

print("Final Predictions:")
for i in range(len(X)):
    prob = final_predictions[i][0]
    binary = binary_predictions[i][0]
    expected = y[i][0]
    status = "✓" if binary == expected else "✗"
    print(f"Input: {X[i]} -> Probability: {prob:.4f} -> Binary: {binary} -> Expected: {expected} {status}")

# Calculate accuracy
accuracy = np.mean(binary_predictions == y) * 100
print(f"\nAccuracy: {accuracy}%")

# Visualize the training process
plt.figure(figsize=(12, 4))

# Plot 1: Loss over time
plt.subplot(1, 2, 1)
plt.plot(nn.loss_history)
plt.title('Training Loss Over Time')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)

# Plot 2: XOR decision boundary visualization
plt.subplot(1, 2, 2)

# Create a mesh grid for visualization
xx, yy = np.meshgrid(np.linspace(-0.5, 1.5, 100), 
                     np.linspace(-0.5, 1.5, 100))
mesh_points = np.c_[xx.ravel(), yy.ravel()]
Z = nn.predict(mesh_points)
Z = Z.reshape(xx.shape)

# Plot decision boundary
plt.contourf(xx, yy, Z, levels=50, alpha=0.8, cmap='RdYlBu')
plt.colorbar(label='Predicted Probability')

# Plot data points
colors = ['red', 'blue']
labels = ['XOR = 0', 'XOR = 1']
for i in range(len(X)):
    color = colors[y[i][0]]
    label = labels[y[i][0]] if i == 0 or (i == 2 and y[i][0] == 1) else ""
    plt.scatter(X[i][0], X[i][1], c=color, s=200, edgecolors='black', linewidth=2, label=label)

plt.title('XOR Decision Boundary')
plt.xlabel('Input 1')
plt.ylabel('Input 2')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("\n=== KEY CONCEPTS DEMONSTRATED ===")
print("1. PERCEPTRON LIMITATION: A single layer cannot solve XOR (not linearly separable)")
print("2. HIDDEN LAYERS: Added non-linearity allows the network to solve XOR")
print("3. FORWARD PROPAGATION: Data flows from input -> hidden -> output")
print("4. ACTIVATION FUNCTIONS: Sigmoid adds non-linearity (essential for XOR)")
print("5. BACKPROPAGATION: Errors propagate backward to update weights")
print("6. GRADIENT DESCENT: Weights updated to minimize loss")
print("7. LEARNING RATE: Controls how big steps the network takes while learning")
print("8. BIAS TERMS: Additional parameters that help the network fit better")

print("\n=== NETWORK ARCHITECTURE ===")
print(f"Input Layer: 2 neurons (for 2 inputs)")
print(f"Hidden Layer: 4 neurons with sigmoid activation")
print(f"Output Layer: 1 neuron with sigmoid activation")
print(f"Total Parameters: {nn.W1.size + nn.b1.size + nn.W2.size + nn.b2.size}")
print(f"  - W1: {nn.W1.shape} = {nn.W1.size} weights")
print(f"  - b1: {nn.b1.shape} = {nn.b1.size} biases") 
print(f"  - W2: {nn.W2.shape} = {nn.W2.size} weights")
print(f"  - b2: {nn.b2.shape} = {nn.b2.size} biases")