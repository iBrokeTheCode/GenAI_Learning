# Overview of the ML Pipeline with PyTorch - Part 2: Models

## 🇪🇸 Spanish

### 1. Construcción del Modelo con `nn.Module`

Aunque `nn.Sequential` es útil, el patrón estándar de PyTorch es crear una clase que herede de **`nn.Module`**. Este patrón ofrece mayor control sobre el flujo de datos.

- **Estructura Base:** Cada clase de modelo debe tener dos métodos:
  - **`__init__` (Constructor):** Se usa para **definir y declarar** las capas del modelo (como `nn.Linear`, `nn.Conv2d`, etc.). Es como reunir las "herramientas".
  - **`forward`:** **Describe el flujo de datos** a través de las capas definidas en `__init__`. Aquí es donde se define el orden real de las operaciones.
- **La Llamada Mágica:** Para ejecutar el modelo (pasar los datos), siempre se llama a la instancia como una función (`model(input_data)`), **nunca** directamente a `model.forward(input_data)`.
  - `model(input_data)` realiza verificaciones internas, rastrea las matemáticas necesarias y configura el sistema para la actualización posterior de los parámetros.
- **`super().__init__` (Necesario):** Esta llamada es crucial. No es solo un _boilerplate_ de Python; **inicializa el sistema de seguimiento** de PyTorch que registra todos los **parámetros aprendibles** (pesos y sesgos). Sin ella, las capas no se registran correctamente.

### 2. El Bucle de Entrenamiento (Training Loop)

El núcleo del entrenamiento es una secuencia estándar y ordenada de operaciones. **El orden es vital**, ya que un cambio puede llevar a fallos silenciosos (el modelo entrena mal sin lanzar un error).

1.  **`optimizer.zero_grad()`:** Borra los cálculos de gradientes antiguos del lote (batch) anterior. Si se omite, los gradientes se acumulan incorrectamente.
2.  **`loss.backward()`:** Calcula los gradientes de la pérdida (_loss_) con respecto a los parámetros del modelo. Esto determina las "mejoras".
3.  **`optimizer.step()`:** **Actualiza** los pesos y sesgos del modelo utilizando los gradientes calculados en el paso 2.

- **Consecuencias de Errores de Orden:**
  - Si se intercambian `backward()` y `step()`, el modelo se actualiza con los gradientes del lote **anterior**.
  - Si se pone `zero_grad()` después de `backward()`, se borra el trabajo de `backward()`.

### 3. Evaluación del Modelo (Evaluation)

Evaluar el rendimiento del modelo en **datos no vistos** es crucial para medir su capacidad de generalización.

- **`model.eval()`:** **Establece el modelo en modo de evaluación.** Esto es necesario porque algunas capas (como _Dropout_ o _BatchNorm_) se comportan de manera diferente durante el entrenamiento que durante la inferencia/evaluación. También es más eficiente computacionalmente.
- **`torch.no_grad()`:** Desactiva el seguimiento interno del historial de cálculos que PyTorch realiza para el _backpropagation_.
  - **Propósito:** Evita el desperdicio de memoria y el riesgo de _crashes_ durante la validación, ya que **no se necesitan gradientes** para la evaluación.
- **Métrica Común (Clasificación):** La **precisión (_Accuracy_)** es la métrica más simple. Se calcula dividiendo el número de predicciones correctas por el número total de intentos.

## 🇬🇧 English

### 1. Model Building with `nn.Module`

While `nn.Sequential` is useful, the standard PyTorch pattern is to create a class that inherits from **`nn.Module`**. This pattern offers greater control over the data flow.

- **Basic Structure:** Every model class must have two methods:
  - **`__init__` (Constructor):** Used to **define and declare** the model's layers (like `nn.Linear`, `nn.Conv2d`, etc.). It's like gathering your "tools."
  - **`forward`:** **Describes the data flow** through the layers defined in `__init__`. This is where the actual order of operations is defined.
- **The Magic Call:** To run the model (pass the data), you always call the instance like a function (`model(input_data)`), **never** directly as `model.forward(input_data)`.
  - `model(input_data)` performs internal checks, tracks necessary math, and sets up the system for subsequent parameter updates.
- **`super().__init__` (Necessary):** This call is crucial. It's not just Python boilerplate; it **initializes PyTorch's tracking system** which registers all **learnable parameters** (weights and biases). Without it, the layers are not properly registered.

### 2. The Training Loop

The core of training is a standard, ordered sequence of operations. **The order is vital**, as a change can lead to silent failure (the model trains incorrectly without throwing an error).

1.  **`optimizer.zero_grad()`:** Clears out old gradient calculations from the previous batch. If skipped, gradients will accumulate incorrectly.
2.  **`loss.backward()`:** Computes the gradients of the loss with respect to the model's parameters. This determines the "improvements."
3.  **`optimizer.step()`:** **Updates** the model's weights and biases using the gradients calculated in step 2.

- **Consequences of Order Errors:**
  - If you swap `backward()` and `step()`, the model updates itself using the gradients from the **previous batch**.
  - If you place `zero_grad()` after `backward()`, all the work done by `backward()` is immediately discarded.

### 3. Model Evaluation

Testing the model's performance on **unseen data** is critical to measure its ability to generalize.

- **`model.eval()`:** **Sets the model into evaluation mode.** This is needed because some layers (like _Dropout_ or _BatchNorm_) behave differently during training than during inference/evaluation. It is also more computationally efficient.
- **`torch.no_grad()`:** Disables the internal tracking of the computational history that PyTorch performs for backpropagation.
  - **Purpose:** Prevents memory wastage and the risk of crashes during validation, as **gradients are not needed** for evaluation.
- **Common Metric (Classification):** **Accuracy** is the simplest metric. It's calculated by dividing the number of correct predictions by the total number of attempts.
