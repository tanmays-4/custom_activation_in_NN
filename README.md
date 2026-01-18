# custom_activation_in_NN
 This project explores the design and testing of a custom activation function inside a small neural network, without using deep learning frameworks.
# Task 2: Custom Activation Function in Neural Network

This project explores the design and testing of a custom activation function inside a small neural network, without using deep learning frameworks.

## Custom Activation Function
A custom activation function called **Soft-Bent** is defined as:

f(x) = x / (1 + |x|)

This function is smooth, bounded between -1 and 1, and reduces saturation compared to Sigmoid.

## Dataset
- IRIS dataset

## Network Setup
- Input layer: 4 neurons  
- Hidden layer: 5 neurons  
- Output layer: 1 neuron  

## Training Details
- Optimizer: Gradient Descent  
- Loss function: Mean Squared Error  
- Epochs: 200  

## Results
Two experiments were performed:
1. Using **Sigmoid** activation
2. Using **Soft-Bent (Custom)** activation

Sigmoid converges faster and reaches a lower final loss.  
The custom activation shows stable but slower convergence, demonstrating learnable behavior.

### Loss Comparison
![Loss Comparison](<img width="468" height="358" alt="loss_comparison" src="https://github.com/user-attachments/assets/74e0b7a1-351e-4da8-a1cd-cc88bfbef273" />
)

### Custom Activation Function Curve
![Activation Function](<img width="454" height="341" alt="activation_function" src="https://github.com/user-attachments/assets/f6353bfc-a25e-4c41-bea0-8fa5590f755b" />
)

## How to Run
```bash
pip install numpy matplotlib scikit-learn
python task2.py
