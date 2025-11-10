# Image Classification - Part 1: Preparing the Data and Building the Model

## 🇪🇸 Spanish

El _dataset_ MNIST (dígitos escritos a mano) es el "hola mundo" de la visión por computadora. Contiene $60.000$ imágenes para entrenamiento y $10.000$ para prueba, todas de $28 \times 28$ píxeles en escala de grises.

### 1. Pre-Procesamiento con `Transforms`

Para preparar las imágenes, se utiliza **TorchVision** (la biblioteca de visión por computadora de PyTorch), que incluye _datasets_ y herramientas de procesamiento.

- **`transforms.ToTensor()`:**
  - Convierte la imagen (ej. de PIL) a un **Tensor** de PyTorch.
  - Escala automáticamente los valores de los píxeles (que originalmente van de $0$ a $255$) al rango $[0, 1]$.
- **`transforms.Normalize(mean, std)`:**
  - **Normaliza** los datos para que se centren alrededor de **cero** y se escalen usando la desviación estándar.
  - Los valores específicos utilizados (ej. $\text{mean}=0.1307, \text{std}=0.3081$) son el promedio y la desviación estándar de **todo el conjunto de entrenamiento de MNIST**. Esto asegura que los datos sean consistentes y acelera el aprendizaje del modelo.

### 2. Carga de Datos con `Dataset` y `DataLoader`

- **`Dataset` (TorchVision):** Se utiliza para cargar y almacenar las imágenes.
  - `train=True/False`: Selecciona el conjunto de entrenamiento ($60.000$ imágenes) o el de prueba ($10.000$ imágenes).
  - `download=True`: Descarga el _dataset_ si no existe localmente.
  - `transform`: Aplica las transformaciones definidas a cada imagen automáticamente al cargarla.
- **`DataLoader`:** Sirve los datos en lotes (_batches_).
  - **Entrenamiento:** `batch_size=64, shuffle=True`. Se utiliza un tamaño de lote moderado y se **mezclan** los datos en cada época para evitar que el modelo aprenda patrones basados en el orden de las muestras (ej. que todos los ceros vengan al principio).
  - **Prueba:** `batch_size=1000, shuffle=False`. Se pueden usar lotes más grandes porque no se calculan gradientes (no se necesita optimización) y **no se mezclan** los datos, ya que el orden no afecta el rendimiento final de la evaluación.

### 3. Arquitectura del Modelo (`nn.Module`)

Se define la clase del modelo heredando de `nn.Module` para obtener control sobre la estructura.

- **Forma de Entrada del Lote:** Un lote de imágenes MNIST llega con la forma $(\text{Batch Size}, \text{Canales}, \text{Alto}, \text{Ancho})$. Ej. $(64, 1, 28, 28)$.
- **La Capa `Flatten` (Aplanamiento):**
  - **Necesidad:** Las **capas lineales** (`nn.Linear`) esperan un **vector plano** de números como entrada, no una cuadrícula 2D.
  - **Función:** `Flatten` toma cada imagen de $1 \times 28 \times 28$ y la **redimensiona** a un vector de $784$ valores (ya que $28 \times 28 = 784$).
  - **Forma de Salida:** El lote se transforma a $(\text{Batch Size}, 784)$. Ej. $(64, 784)$.
- **Estructura del Modelo (Sequential):**
  1.  **`Flatten()`:** Aplanar la imagen.
  2.  **`nn.Linear(784, 128)`:** Capa oculta. Toma $784$ píxeles y genera $128$ características ocultas.
  3.  **`nn.ReLU()`:** Función de activación (introduce no linealidad).
  4.  **`nn.Linear(128, 10)`:** Capa de salida. Toma $128$ características y genera $10$ valores de salida (uno por cada dígito/clase de 0 a 9).

## 🇬🇧 English

The MNIST dataset (handwritten digits) is the "hello world" of computer vision. It contains $60,000$ training images and $10,000$ test images, all $28 \times 28$ pixels in grayscale.

### 1. Pre-processing with `Transforms`

To prepare the images, **TorchVision** (PyTorch's computer vision library) is used, which includes popular datasets and processing tools.

- **`transforms.ToTensor()`:**
  - Converts the image (e.g., from PIL) to a PyTorch **Tensor**.
  - Automatically **scales** the pixel values (originally $0$ to $255$) to the range $[0, 1]$.
- **`transforms.Normalize(mean, std)`:**
  - **Normalizes** the data to be centered around **zero** and scaled using the standard deviation.
  - The specific values used (e.g., $\text{mean}=0.1307, \text{std}=0.3081$) are the mean and standard deviation of the **entire MNIST training set**. This ensures data consistency and speeds up model learning.

### 2. Data Loading with `Dataset` and `DataLoader`

- **`Dataset` (TorchVision):** Used to load and store images.
  - `train=True/False`: Selects the training set ($60,000$ images) or the test set ($10,000$ images).
  - `download=True`: Downloads the dataset if it's not present locally.
  - `transform`: Applies the defined transformations to each image automatically upon loading.
- **`DataLoader`:** Serves the data in batches.
  - **Training:** `batch_size=64, shuffle=True`. A moderate batch size is used, and data is **shuffled** in every epoch to prevent the model from learning patterns based on the samples' order (e.g., all zeros coming first).
  - **Testing:** `batch_size=1000, shuffle=False`. Larger batches can be used because no gradients are calculated (no optimization needed), and the data is **not shuffled**, as order doesn't affect the final evaluation performance.

### 3. Model Architecture (`nn.Module`)

The model class is defined by inheriting from `nn.Module` to gain control over the structure.

- **Batch Input Shape:** A batch of MNIST images arrives with the shape $(\text{Batch Size}, \text{Channels}, \text{Height}, \text{Width})$. E.g., $(64, 1, 28, 28)$.
- **The `Flatten` Layer:**
  - **Necessity:** **Linear layers** (`nn.Linear`) expect a **flat vector** of numbers as input, not a 2D grid.
  - **Function:** `Flatten` takes each $1 \times 28 \times 28$ image and **reshapes** it into a vector of $784$ values ($28 \times 28 = 784$).
  - **Output Shape:** The batch is transformed to $(\text{Batch Size}, 784)$. E.g., $(64, 784)$.
- **Model Structure (Sequential):**
  1.  **`Flatten()`:** Flattens the image.
  2.  **`nn.Linear(784, 128)`:** Hidden layer. Takes $784$ pixels and generates $128$ hidden features.
  3.  **`nn.ReLU()`:** Activation function (introduces non-linearity).
  4.  **`nn.Linear(128, 10)`:** Output layer. Takes $128$ features and outputs $10$ values (one for each digit/class from 0 to 9).
