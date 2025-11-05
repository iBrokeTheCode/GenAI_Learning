# Overview of the ML Pipeline with Pytorch - Part 1: Data

## 🇪🇸 Spanish

### **El Problema del Big Data**

- Al trabajar con _datasets_ grandes (millones de registros o imágenes), cargar todos los datos simultáneamente en la **RAM** del ordenador es inviable. Esto provocaría que el sistema se quede sin memoria y colapse.
- **La Solución:** Trabajar con los datos en **lotes (_batches_)**, que son porciones pequeñas y manejables del _dataset_ completo.

### **Las Tres Utilidades Clave de PyTorch**

PyTorch proporciona tres herramientas principales que trabajan juntas para procesar, formatear y servir los datos de manera eficiente durante el entrenamiento: **`Transforms`**, **`Dataset`** y **`DataLoader`**.

#### 1. Transforms (Transformaciones)

- Son **operaciones** que se aplican a **cada punto de dato** individualmente mientras se carga, preparando el dato para el modelo.
- **`Compose`**: Permite encadenar múltiples transformaciones para que se ejecuten en orden.
- **Transformaciones Comunes:**
  - **`ToTensor`**: Convierte los datos al formato **Tensor** de PyTorch y los **escala** automáticamente al rango $[0, 1]$.
  - **`Normalize`**: Ajusta los valores aún más, **centrándolos alrededor de cero** y escalándolos usando la desviación estándar.
- **Propósito:** Las redes neuronales entrenan mejor cuando las entradas son números pequeños, centrados cerca de cero. Estas transformaciones aseguran esta condición.

#### 2. Dataset (Conjunto de Datos)

- Es una clase que encapsula cómo **obtener una muestra** del disco cuando se le solicita. Es la clave para manejar _datasets_ masivos, ya que **no precarga** todos los datos.
- **Funciones Principales:** Sabe dónde reside el dato, cómo cargar una muestra específica (mediante indexación), cuántas muestras totales hay, y cómo aplicar las `Transforms` a la muestra mientras se carga.
- **Parámetros Comunes:** Define la ubicación del dato, si se usa la versión de **entrenamiento** (`train=True`) o **prueba** (_test_), y si se debe **descargar** (`download=True`) si no existe localmente.

#### 3. DataLoader (Cargador de Datos)

- Es la parte final del _pipeline_ que se encarga de **servir los datos en lotes (_batches_)** al modelo.
- **Mecanismo:** Solicita un _batch_ a la vez al objeto `Dataset`.
- **Parámetros Clave:**
  - **`batch_size`**: Define cuántas muestras se incluirán en cada lote.
  - **`shuffle=True`**: Mezcla los datos en cada época, lo que ayuda al modelo a aprender de forma más efectiva y evita sesgos.

### **El Flujo Completo**

El patrón completo asegura un manejo eficiente de la memoria: **`Dataset`** obtiene muestras individuales de forma perezosa $\rightarrow$ **`Transforms`** las preparan $\rightarrow$ **`DataLoader`** las agrupa en _batches_ listos para el entrenamiento.

## 🇬🇧 English

### **The Big Data Problem**

- When working with large datasets (millions of records or images), loading all data into the computer's **RAM** simultaneously is infeasible. This would lead to the system running out of memory and crashing.
- **The Solution:** Work with data in **batches**, which are small, manageable chunks of the full dataset.

### **PyTorch's Three Core Utilities**

PyTorch provides three main tools that work together to efficiently process, format, and serve data during training: **`Transforms`**, **`Dataset`**, and **`DataLoader`**.

#### 1. Transforms

- These are **operations** applied to **each individual data point** as it is loaded, preparing the data for the model.
- **`Compose`**: Allows multiple transformations to be chained and executed in sequence.
- **Common Transforms:**
  - **`ToTensor`**: Converts the data into the PyTorch **Tensor** format and automatically **scales** it to the range $[0, 1]$.
  - **`Normalize`**: Further adjusts these values by **centering them around zero** and scaling them using the standard deviation.
- **Purpose:** Neural networks train much better when inputs are small numbers, ideally centered close to zero. These transforms ensure this condition.

#### 2. Dataset

- This is a class that encapsulates how to **fetch a single sample** from the disk when requested. It is the key to handling massive datasets because it **does not preload** all data.
- **Core Functions:** It manages where the data resides, how to load a specific sample (via indexing), the total number of samples, and how to apply the `Transforms` to the sample as it is loaded.
- **Common Parameters:** Defines the data location, whether the **training** (`train=True`) or **test** set is used, and whether the data should be **downloaded** (`download=True`) if not present locally.

#### 3. DataLoader

- This is the final part of the pipeline responsible for **serving data in batches** to the model.
- **Mechanism:** It requests one batch at a time from the `Dataset` object.
- **Key Parameters:**
  - **`batch_size`**: Defines how many samples are included in each batch.
  - **`shuffle=True`**: Shuffles the data in each epoch, which helps the model learn more effectively and prevents biases.

### **The Complete Flow**

The complete pattern ensures efficient memory handling: **`Dataset`** fetches individual samples lazily $\rightarrow$ **`Transforms`** prepare them $\rightarrow$ **`DataLoader`** groups them into batches ready for training.
