# Loss

## 🇪🇸 spanish

El ciclo de entrenamiento se resume en: **Medir** (Pérdida), **Diagnosticar** (`backward`), y **Actualizar** (`step`). La función de pérdida se encarga del primer paso.

### 1. El Concepto de Pérdida

- **Función de Pérdida (Loss Function) / Criterio:** Es una métrica que **compara** las predicciones del modelo con las respuestas verdaderas (etiquetas).
- **Resultado:** Produce un único número que resume todos los errores cometidos por el modelo en un lote (_batch_). **Un número más alto significa un mayor error** y peor rendimiento.
- **Objetivo Final:** El proceso de entrenamiento siempre busca **minimizar** la pérdida.

### 2. Pérdida por Error Cuadrático Medio (MSE)

El **Error Cuadrático Medio** (_Mean Squared Error_ o **MSE**) es ideal para problemas de **Regresión**, donde se predicen valores continuos (como tiempo, temperatura o precio).

- **Cálculo:** Se calcula la diferencia entre la predicción y el valor real (el error), se eleva **al cuadrado** para eliminar el signo negativo, y finalmente se promedian estos errores cuadrados.
- **Fórmula (Conceptual):** $\text{MSE} = \frac{1}{N} \sum_{i=1}^{N} (y_{\text{predicha}, i} - y_{\text{real}, i})^2$.
- **Beneficios del Cuadrado:**
  1.  **Evita la Cancelación:** Al cuadrar el error, los errores positivos y negativos no se anulan al promediar (evitando un `Loss = 0` engañoso).
  2.  **Castigo Mayor a Errores Grandes:** El cuadrado penaliza los errores grandes de forma desproporcionada, lo que anima al modelo a corregir las peores predicciones primero.

### 3. Pérdida de Entropía Cruzada (Cross-Entropy Loss)

La **Pérdida de Entropía Cruzada** (_Cross-Entropy Loss_) es la función estándar para problemas de **Clasificación**, donde se predice una categoría (como un dígito, animal o palabra).

- **Mecanismo de Salida:** Para la clasificación, el modelo no solo elige una respuesta, sino que produce una **puntuación de confianza (probabilidad)** para cada clase posible (las puntuaciones deben sumar 1 o 100%).
- **Principio Clave:** La Entropía Cruzada **castiga la confianza excesiva en respuestas incorrectas.**
  - Si el modelo está $95\%$ seguro de que es 'A' y es incorrecto $\rightarrow$ **Pérdida MUY alta.**
  - Si el modelo solo está $55\%$ seguro de que es 'A' y es incorrecto $\rightarrow$ **Pérdida más baja.**
- **Objetivo:** Modela el comportamiento para que sea **confiado** con las respuestas correctas e **inseguro** con las incorrectas.

### **Advertencia Importante**

- **No Mezclar:** Usar MSE para clasificación o Entropía Cruzada para regresión puede llevar a un entrenamiento inestable, lento o simplemente roto, ya que cada función espera un tipo de salida específico (valores continuos vs. distribuciones de probabilidad).
- **No Comparar Números Crudos:** Los valores absolutos de pérdida de diferentes funciones no son directamente comparables (ej. MSE de 0.08 vs. Entropía Cruzada de 2.3). Solo importa que el número de la función elegida **disminuya** durante el entrenamiento.

## 🇬🇧 English

The training cycle is summarized as: **Measure** (Loss), **Diagnose** (`backward`), and **Update** (`step`). The Loss Function handles the first step.

### 1. The Loss Concept

- **Loss Function / Criterion:** This is a metric that **compares** the model's predictions to the true answers (labels).
- **Output:** It produces a single number that summarizes all the mistakes the model made in a batch. **A higher number signifies a greater error** and worse performance.
- **Ultimate Goal:** The training process always aims to **minimize** the loss.

### 2. Mean Squared Error Loss (MSE)

**Mean Squared Error (MSE)** is ideal for **Regression** problems, where continuous values are predicted (such as time, temperature, or price).

- **Calculation:** The difference between the prediction and the true value (the error) is calculated, then **squared** to remove the negative sign, and finally, these squared errors are averaged.
- **Formula (Conceptual):** $\text{MSE} = \frac{1}{N} \sum_{i=1}^{N} (y_{\text{predicted}, i} - y_{\text{actual}, i})^2$.
- **Benefits of Squaring:**
  1.  **Prevents Cancellation:** By squaring the error, positive and negative errors don't cancel each other out when averaging (preventing a misleading `Loss = 0`).
  2.  **Heavier Penalty for Larger Errors:** Squaring disproportionately penalizes large mistakes, encouraging the model to fix the worst predictions first.

### 3. Cross-Entropy Loss

**Cross-Entropy Loss** is the standard function for **Classification** problems, where a category is predicted (like a digit, animal, or word).

- **Output Mechanism:** For classification, the model doesn't just pick one answer; it outputs a **confidence score (probability)** for every possible class (scores should sum to 1 or 100%).
- **Key Principle:** Cross-Entropy Loss **punishes overconfidence in wrong answers.**
  - If the model is $95\%$ sure it's 'A' and it's incorrect $\rightarrow$ **VERY high Loss.**
  - If the model is only $55\%$ sure it's 'A' and it's incorrect $\rightarrow$ **Lower Loss.**
- **Goal:** Shapes the model's behavior to be **confident** in the correct answers and **unsure** about the incorrect ones.

### **Important Warning**

- **Do Not Mix:** Using MSE for classification or Cross-Entropy for regression can lead to unstable, slow, or outright broken training, as each function expects a specific type of output (continuous values vs. probability distributions).
- **Do Not Compare Raw Numbers:** The absolute loss values from different functions are not directly comparable (e.g., MSE of 0.08 vs. Cross-Entropy of 2.3). The only thing that matters is that the number for the chosen function **decreases** during training.
