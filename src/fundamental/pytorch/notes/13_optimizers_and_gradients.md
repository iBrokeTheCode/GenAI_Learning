# Optimizers and Gradients

## 🇪🇸 Spanish

Después de medir la pérdida, el ciclo de entrenamiento procede con el diagnóstico (`backward`) y la actualización (`step`).

### 1. Los Gradientes (Diagnóstico con `backward()`)

- **Rol de `backward()`:** Esta función actúa como un "detective" del modelo. Mira la pérdida calculada y rastrea **hacia atrás** a través de la red neuronal para determinar cómo cada peso y sesgo contribuyó al error.
- **Gradientes (Scores Diagnósticos):** El resultado de `backward()` son los **gradientes**, que son puntuaciones diagnósticas adjuntas a cada parámetro entrenable.
  - **Magnitud:** El valor absoluto indica cuánto influyó un peso en el error.
  - **Signo:**
    - Valor **positivo** $\rightarrow$ Aumentar el peso empeoraría la pérdida.
    - Valor **negativo** $\rightarrow$ Aumentar el peso habría ayudado a reducir la pérdida.
- **Concepto Erróneo:** **`backward()` solo calcula los gradientes.** No actualiza los pesos; la actualización ocurre más tarde.

### 2. El Descenso de Gradiente (Gradient Descent)

El objetivo es minimizar la pérdida. Conceptualmente, es como buscar el fondo de un valle.

- **Principio:** El gradiente indica la dirección de la pendiente (cuesta arriba). Para minimizar la pérdida, debemos movernos en la **dirección opuesta al gradiente** (cuesta abajo).
- **Actualización de Peso:** Se realiza una corrección al peso proporcional a la magnitud del gradiente, pero en la dirección opuesta:
  $$\text{Peso}_{\text{nuevo}} = \text{Peso}_{\text{viejo}} - (\text{Tasa de Aprendizaje} \times \text{Gradiente})$$

### 3. El Optimizador (`optimizer.step()`)

El optimizador es la clase que implementa la estrategia de Descenso de Gradiente y realiza la actualización de los parámetros.

- **Tasa de Aprendizaje (_Learning Rate_):** Es un hiperparámetro crucial que **escala** el tamaño de la corrección del peso.
  - _Tasa muy pequeña:_ Progreso lento, puede tardar una eternidad.
  - _Tasa muy grande:_ Pasos gigantes que pueden rebotar o "saltar" el punto mínimo (_overshooting_).
  - _Tasa adecuada:_ Progreso constante hacia el mínimo.
- **Optimizadores Comunes:**
  - **SGD (_Stochastic Gradient Descent_):** El enfoque más simple. Aplica correcciones uniformes basadas en el gradiente.
  - **Adam:** Un optimizador más avanzado que **adapta** la tasa de aprendizaje para **cada peso individualmente** (como tener un "asistente inteligente"). Es la opción predeterminada y a menudo más rápida para empezar.

### 4. La Importancia de `optimizer.zero_grad()`

- **Acumulación por Defecto:** PyTorch está diseñado para **acumular gradientes** cada vez que se llama a `backward()`. Los nuevos gradientes se **suman** a los gradientes existentes.
- **Problema Común:** Si no se llama a `zero_grad()` al comienzo de cada lote de entrenamiento, los diagnósticos de los lotes anteriores se suman incorrectamente, haciendo que las actualizaciones de `step()` sean masivas e incorrectas, lo que "rompe" el entrenamiento.
- **Uso:** Debe llamarse **al inicio de cada ciclo de entrenamiento** para garantizar que solo se calculen los gradientes para el lote actual.

## 🇬🇧 English

After measuring the loss, the training cycle proceeds with diagnosis (`backward`) and updating (`step`).

### 1. Gradients (Diagnosis with `backward()`)

- **Role of `backward()`:** This function acts as the model's "detective." It looks at the calculated loss and traces **backward** through the neural network to determine how each weight and bias contributed to the error.
- **Gradients (Diagnostic Scores):** The result of `backward()` are the **gradients**, which are diagnostic scores attached to every trainable parameter.
  - **Magnitude:** The absolute value indicates how much a weight influenced the error.
  - **Sign:**
    - **Positive** value $\rightarrow$ Increasing the weight would make the loss worse.
    - **Negative** value $\rightarrow$ Increasing the weight would have helped reduce the loss.
- **Common Misconception:** **`backward()` only calculates the gradients.** It does not update the weights; the actual update happens later.

### 2. Gradient Descent

The goal is to minimize the loss. Conceptually, it's like searching for the bottom of a valley.

- **Principle:** The gradient indicates the direction of the slope (uphill). To minimize loss, we must move in the **opposite direction of the gradient** (downhill).
- **Weight Update:** A correction is made to the weight, proportional to the magnitude of the gradient, but in the opposite direction:
  $$\text{Weight}_{\text{new}} = \text{Weight}_{\text{old}} - (\text{Learning Rate} \times \text{Gradient})$$

### 3. The Optimizer (`optimizer.step()`)

The optimizer is the class that implements the Gradient Descent strategy and performs the parameter update.

- **Learning Rate (LR):** This is a crucial hyperparameter that **scales** the size of the weight correction.
  - _Very Small LR:_ Slow progress, may take forever to reach the bottom.
  - _Very Large LR:_ Giant leaps that can bounce back and forth or "overshoot" the minimum point.
  - _Optimal LR:_ Steady progress towards the minimum.
- **Common Optimizers:**
  - **SGD (_Stochastic Gradient Descent_):** The simplest approach. Applies uniform corrections based on the gradient.
  - **Adam:** A more advanced optimizer that **adapts** the learning rate for **each individual weight** (like having a "smart assistant"). It is the reliable and often faster default choice.

### 4. The Importance of `optimizer.zero_grad()`

- **Default Accumulation:** PyTorch is designed to **accumulate gradients** every time `backward()` is called. New gradients are **added** to existing ones.
- **The Common Problem:** If you don't call `zero_grad()` at the start of every training batch, the diagnostics from previous batches are incorrectly summed up, causing the `step()` updates to be massive and wrong, which effectively "breaks" the training.
- **Usage:** Must be called **at the start of every training loop** to ensure gradients are calculated only for the current batch.
