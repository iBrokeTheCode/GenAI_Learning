# Activation Functions

## 🇪🇸 Spanish

- **El Problema de la Linealidad:** Aunque una **Red Neuronal (RN)** tenga muchas capas (una **Red Neuronal Profunda**), sin una función de activación, la composición de transformaciones lineales (multiplicación por el peso y suma del sesgo: $W \cdot X + b$) sigue siendo inherentemente **lineal**. Esto significa que la RN solo podría resolver problemas de clasificación linealmente separables (algo muy limitado).
- **La Solución: Introducir No Linealidad:** Para que una RN pueda aprender **patrones complejos y no lineales**, es necesario introducir las **Funciones de Activación** (FA).
- **Ubicación y Propósito:** Una FA se aplica **después** de la operación de combinación lineal ($Z = W \cdot X + b$) en cada neurona. Su objetivo es transformar la salida de la neurona, introduciendo la **no linealidad** esencial para el aprendizaje profundo.
- **Funciones Comunes:**

  - **Sigmoide ($\sigma$):** Comprime la salida a un rango entre 0 y 1. Históricamente popular, pero con problemas de "desvanecimiento del gradiente" (_vanishing gradient_).
  - **Tanh (Tangente Hiperbólica):** Comprime la salida a un rango entre -1 y 1. Es una versión centrada en cero de Sigmoide, mejorando un poco el gradiente.
  - **ReLU (Rectified Linear Unit):** Es la función más utilizada actualmente por su **eficiencia computacional** y por mitigar el problema del _vanishing gradient_.

- **Definición de ReLU:**
  - Si la entrada ($Z$) es **negativa** ($Z < 0$), la salida es **cero** (0).
  - Si la entrada ($Z$) es **positiva** ($Z \geq 0$), la salida es **el mismo valor** ($Z$).
  - Matemáticamente: $\text{ReLU}(Z) = \max(0, Z)$.

## 🇬🇧 English

- **The Linearity Problem:** Even if a **Neural Network (NN)** has many layers (a **Deep Neural Network**), without an activation function, the composition of linear transformations (weight multiplication plus bias addition: $W \cdot X + b$) remains inherently **linear**. This means the NN could only solve linearly separable classification problems (a very limited scope).
- **The Solution: Introducing Non-Linearity:** For an NN to be able to learn **complex, non-linear patterns**, it's necessary to introduce **Activation Functions** (AFs).
- **Placement and Purpose:** An AF is applied **after** the linear combination operation ($Z = W \cdot X + b$) in each neuron. Its goal is to transform the neuron's output, introducing the **non-linearity** that is essential for deep learning.
- **Common Functions:**
  - **Sigmoid ($\sigma$):** Compresses the output to a range between 0 and 1. Historically popular, but prone to the "vanishing gradient" problem.
  - **Tanh (Hyperbolic Tangent):** Compresses the output to a range between -1 and 1. It's a zero-centered version of Sigmoid, slightly improving the gradient.
  - **ReLU (Rectified Linear Unit):** This is the most widely used function today due to its **computational efficiency** and its ability to mitigate the _vanishing gradient_ problem.
- **Definition of ReLU:**
  - If the input ($Z$) is **negative** ($Z < 0$), the output is **zero** (0).
  - If the input ($Z$) is **positive** ($Z \geq 0$), the output is **the same value** ($Z$).
  - Mathematically: $\text{ReLU}(Z) = \max(0, Z)$.
