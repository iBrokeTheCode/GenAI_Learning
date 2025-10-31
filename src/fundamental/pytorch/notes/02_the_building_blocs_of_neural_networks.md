# The Building Block of Neural Networks

## 🇪🇸 Spanish

- Una neurona tiene entradas (inputs), salidas (outputs) y parámetros, que son el peso (W) y el sesgo (b).
- La neurona toma los valores de entrada, los multiplica por su peso correspondiente y suma el sesgo:

  $$ y = W \cdot x + b $$

- Los parámetros (W y b) se inicializan aleatoriamente al comienzo del entrenamiento y luego se ajustan comparando la predicción con el resultado real, calculando así el error. Este proceso se realiza mediante técnicas matemáticas de optimización (por ejemplo, _backpropagation_ y _gradient descent_).
- Cuando combinas muchas neuronas obtienes una red neuronal. Es importante destacar que, ya sea con una o muchas neuronas, cada neurona representa una función lineal; la no linealidad se introduce posteriormente mediante funciones de activación.
- En una red neuronal hay 3 tipos de capas: Capa de entrada, capa oculta y capa de salida.

## 🇬🇧 English

- A neuron has inputs, outputs, and parameters, specifically the weight (W) and the bias (b).
- The neuron takes the input values, multiplies them by their corresponding weights, and adds the bias:

  $$ y = W \cdot x + b $$

- These parameters (W and b) are initialized randomly at the beginning of training and are later adjusted by comparing the prediction with the actual result, calculating the error. This adjustment is performed using mathematical optimization techniques such as backpropagation and gradient descent.
- When many neurons are combined, they form a neural network. It's improtant to note that each individual neuron represents a linear function, non-linearity is introduced later through activation functions.
- In a neural network there are 3 types of layers: Input layer, Hidden layer and Output layer.
