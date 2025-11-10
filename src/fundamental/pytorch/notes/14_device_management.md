# Device Management

## 🇪🇸 Spanish

El **manejo de dispositivos** es esencial en PyTorch. Cada **tensor** y **parámetro del modelo** debe residir en un dispositivo específico (CPU, GPU u otro acelerador). Si los tensores que interactúan no están en el mismo dispositivo, el código fallará con un error común de _Device Mismatch_.

### 1\. CPU vs. GPU (Aceleradores)

- **CPU (Central Processing Unit):** El dispositivo predeterminado. Es de propósito general y procesa operaciones secuencialmente.
- **GPU (Graphics Processing Unit):** Un **acelerador** que procesa operaciones de tensores mucho más rápido (típicamente de 10 a 15 veces más rápido) porque las ejecuta en paralelo.
  - La tecnología más común para GPUs NVIDIA en PyTorch es **CUDA**.

### 2\. Configuración y Elección del Dispositivo

- **Verificar Disponibilidad:** Se comprueba si PyTorch puede usar una GPU: `torch.cuda.is_available()`.
- **Patrón Seguro de Elección:** Se define el dispositivo para usar la GPU si está disponible; de lo contrario, se usa la CPU.
  ```python
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  ```

### 3\. Mover Datos y Modelo al Dispositivo

**PyTorch no mueve los datos automáticamente; debes hacerlo manualmente.**

| Elemento              | ¿Cuándo se mueve?               | Sintaxis                     | Nota Clave                                                                          |
| :-------------------- | :------------------------------ | :--------------------------- | :---------------------------------------------------------------------------------- |
| **Modelo**            | Una sola vez, al crearlo.       | `model.to(device)`           | Mueve todos los pesos y sesgos del modelo.                                          |
| **Datos (Batch)**     | Dentro del bucle, en cada lote. | `data = data.to(device)`     | Debe reasignarse, ya que `.to()` crea un **nuevo** tensor.                          |
| **Etiquetas/Targets** | Dentro del bucle, en cada lote. | `target = target.to(device)` | Las etiquetas también son tensores y deben coincidir con el dispositivo del modelo. |

- **Verificación:** Puedes revisar la ubicación de un tensor: `tensor.device`. Para un modelo, revisa la ubicación de uno de sus parámetros, por ejemplo: `model.layer_name.weight.device`.

### 4\. Gestión de Memoria de la GPU

La memoria de la GPU es **limitada**.

- **Error Común:** Si el modelo y el tamaño del lote (_batch size_) exceden la memoria disponible, se produce un error de "out of memory".
- **`batch_size` Importa:** Un **`batch_size` demasiado grande** es la causa más común de errores de memoria en la GPU.
- **Solución:** Si recibes un error de memoria, la primera solución es **reducir el tamaño del lote** (un buen punto de partida suele ser 32 o 64).

## 🇬🇧 English

**Device management** is essential in PyTorch. Every **tensor** and **model parameter** must reside on a specific device (CPU, GPU, or other accelerator). If interacting tensors are not on the same device, the code will crash with a common **Device Mismatch** error.

### 1\. CPU vs. GPU (Accelerators)

- **CPU (Central Processing Unit):** The default device. It's general-purpose and processes operations sequentially.
- **GPU (Graphics Processing Unit):** An **accelerator** that processes tensor operations much faster (typically 10-15x faster) by executing them in parallel.
  - The most common technology for NVIDIA GPUs in PyTorch is **CUDA**.

### 2\. Setup and Device Selection

- **Check Availability:** You check if PyTorch can use a GPU: `torch.cuda.is_available()`.
- **Safe Selection Pattern:** Define the device to use the GPU if available; otherwise, use the CPU.
  ```python
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  ```

### 3\. Moving Data and Model to Device

**PyTorch will not move data for you automatically; you must do it manually.**

| Element            | When to Move?                     | Syntax                       | Key Note                                                     |
| :----------------- | :-------------------------------- | :--------------------------- | :----------------------------------------------------------- |
| **Model**          | Once, upon creation.              | `model.to(device)`           | Moves all the model's weights and biases.                    |
| **Data (Batch)**   | Inside the loop, for every batch. | `data = data.to(device)`     | **Must be reassigned,** as `.to()` creates a **new** tensor. |
| **Labels/Targets** | Inside the loop, for every batch. | `target = target.to(device)` | Labels are also tensors and must match the model's device.   |

- **Verification:** You can check a tensor's location: `tensor.device`. For a model, check the location of one of its parameters, e.g., `model.layer_name.weight.device`.

### 4\. GPU Memory Management

GPU memory is **limited**.

- **Common Error:** If the model and the **batch size** exceed the available memory, an "out of memory" error will occur.
- **Batch Size Matters:** An **overly large `batch_size`** is the most common cause of GPU memory errors.
- **Fix:** If you get a memory error, the first solution is to **lower your batch size** (a common starting point is 32 or 64).
