# Image Classification - Part 2: Training and Evaluating the Model

## 🇪🇸 Spanish

### 1. Configuración Inicial

1.  **Dispositivo:** Seleccionar `cuda` (GPU) si está disponible; si no, `cpu`.
2.  **Modelo:** Instanciar el modelo (que ya incluye la capa `Flatten` y las capas lineales) y moverlo al `device` seleccionado: `model.to(device)`.
3.  **Función de Pérdida:** Se usa **`nn.CrossEntropyLoss()`** (ideal para clasificación).
4.  **Optimizador:** Se usa **`optim.Adam`** con una tasa de aprendizaje (LR) de $0.001$. Adam es preferido por su capacidad de adaptar el LR, lo que acelera y estabiliza el entrenamiento.

### 2. Función de Entrenamiento por Época (`train()`)

Esta función recorre el _dataset_ de entrenamiento una sola vez.

- **Preparación:**
  - **`model.train()`:** Pone el modelo en modo de entrenamiento (esencial para capas como _Dropout_ y _BatchNorm_).
  - Se inicializan variables para rastrear la pérdida y las predicciones correctas.
- **Bucle del Lote (Batch Loop):** Por cada lote del `DataLoader`:
  1.  **Mover a Dispositivo:** Mover los datos y las etiquetas (`targets`) al `device`.
  2.  **Borrar Gradientes:** **`optimizer.zero_grad()`** (evita la acumulación de gradientes de lotes anteriores).
  3.  **Paso Adelante (_Forward Pass_):** `output = model(data)` (obtener las predicciones).
  4.  **Calcular Pérdida:** `loss = loss_fn(output, target)`.
  5.  **Paso Atrás (_Backward Pass_):** **`loss.backward()`** (calcular los gradientes).
  6.  **Actualizar Pesos:** **`optimizer.step()`** (ajustar los parámetros del modelo).
- **Monitoreo:** Se rastrea la **pérdida** (debe disminuir) y la **precisión** (debe aumentar) a lo largo de la época.

### 3. Función de Evaluación (`test()`)

Esta función prueba el modelo con datos no vistos (el _dataset_ de prueba).

- **Diferencias con Entrenamiento:**
  1.  **`model.eval()`:** Pone el modelo en modo de evaluación.
  2.  **`with torch.no_grad():`:** Desactiva el cálculo y el seguimiento de gradientes. Esto ahorra memoria y acelera la inferencia, ya que no se necesita `backward()`.
- **Proceso:**
  - Se recorren los lotes del `test_dataloader` (sin `shuffle`).
  - Se calcula la predicción (el índice de la puntuación más alta: `output.max(1)[1]`).
  - Se cuenta cuántas predicciones coinciden con las etiquetas reales.
  - Al final, se devuelve la precisión general.

### 4. Bucle Principal

- El modelo se entrena durante un número fijo de épocas (ej. 10).
- Después de cada época de entrenamiento (`train()`), se realiza la evaluación (`test()`) para verificar la **capacidad de generalización** del modelo.
- Si la precisión en el _test set_ deja de mejorar, es una señal de que el modelo ha aprendido lo suficiente (o está empezando a memorizar el _training set_).

## 🇬🇧 English

### 1. Initial Setup

1.  **Device:** Select `cuda` (GPU) if available; otherwise, `cpu`.
2.  **Model:** Instantiate the model (which already includes the `Flatten` layer and linear layers) and move it to the selected `device`: `model.to(device)`.
3.  **Loss Function:** **`nn.CrossEntropyLoss()`** is used (ideal for classification).
4.  **Optimizer:** **`optim.Adam`** is used with a Learning Rate (LR) of $0.001$. Adam is preferred for its ability to adapt the LR, which speeds up and stabilizes training.

### 2. Training Function Per Epoch (`train()`)

This function iterates over the training dataset a single time.

- **Preparation:**
  - **`model.train()`:** Sets the model to training mode (essential for layers like _Dropout_ and _BatchNorm_).
  - Variables are initialized to track running loss and correct predictions.
- **Batch Loop:** For every batch from the `DataLoader`:
  1.  **Move to Device:** Move both the data and the labels (`targets`) to the `device`.
  2.  **Clear Gradients:** **`optimizer.zero_grad()`** (prevents gradient accumulation from previous batches).
  3.  **Forward Pass:** `output = model(data)` (get predictions).
  4.  **Compute Loss:** `loss = loss_fn(output, target)`.
  5.  **Backward Pass:** **`loss.backward()`** (compute gradients).
  6.  **Update Weights:** **`optimizer.step()`** (adjust model parameters).
- **Monitoring:** The **loss** (should decrease) and **accuracy** (should increase) are tracked throughout the epoch.

### 3. Evaluation Function (`test()`)

This function tests the model on unseen data (the test dataset).

- **Differences from Training:**
  1.  **`model.eval()`:** Sets the model to evaluation mode.
  2.  **`with torch.no_grad():`:** Disables gradient calculation and tracking. This saves memory and speeds up inference, as `backward()` is not needed.
- **Process:**
  - Batches from the `test_dataloader` are iterated (no `shuffle`).
  - The prediction (the index of the highest score: `output.max(1)[1]`) is calculated.
  - The number of predictions matching the true labels is counted.
  - The overall accuracy percentage is returned at the end.

### 4. Main Loop

- The model is trained for a fixed number of epochs (e.g., 10).
- After each training epoch (`train()`), evaluation (`test()`) is performed to check the model's **generalization ability**.
- If the accuracy on the test set stops improving, it signals that the model has learned enough (or is starting to memorize the _training set_).
