# Tensors

## 🇪🇸 Spanish

Los **tensores** son la estructura de datos fundamental utilizada en PyTorch para representar y manipular los datos. Son esencialmente **generalizaciones multidimensionales de escalares (0D), vectores (1D) y matrices (2D)**.

- **Representación de Datos:** Los tensores son el formato que PyTorch espera para todas las operaciones, desde los _inputs_ (imágenes, textos) hasta los parámetros del modelo (pesos y sesgos).
- **Inspección de la Forma (`size`/`shape`):** El atributo `size()` (o `shape`) nos permite visualizar la **forma** o dimensiones del tensor.
  - Para un caso común de entrenamiento, como un tensor 2D:
    - El **primer parámetro** (dimensión 0) suele ser el **tamaño del lote** (_batch size_), que indica cuántas muestras se procesan simultáneamente.
    - El **segundo parámetro** (dimensión 1) es el **número de características** (_features_) o el tamaño de la muestra individual.
- **Tipos de Datos (`dtype`):** Aunque PyTorch puede inferir el tipo de dato, es posible (y a menudo necesario) especificar explícitamente el tipo (e.g., `torch.float32`, `torch.int64`).
  - **Casting Automático (Promoción de Tipo):** PyTorch puede realizar una **promoción de tipo** (conversión o _casting_) cuando se realizan operaciones entre tensores con diferentes tipos de datos, eligiendo el tipo más preciso.
- **Conceptos Relacionados:**
  - **Remodelación (_Reshaping_):** Consiste en **cambiar la forma** de un tensor (sus dimensiones), manteniendo el número total de elementos. Esto se hace con funciones como `view()` o `reshape()`. Es crucial, por ejemplo, para aplanar una imagen antes de pasarla a una capa lineal.
  - **Indexación (_Indexing_):** Permite **acceder o modificar** elementos o subconjuntos de un tensor mediante sus índices, de manera similar a como se hace con los _arrays_ o listas en Python (incluyendo el uso de _slicing_ o rebanado).

## 🇬🇧 English

**Tensors** are the foundational data structure used in PyTorch to represent and manipulate data. They are essentially **multi-dimensional generalizations of scalars (0D), vectors (1D), and matrices (2D)**.

- **Data Representation:** Tensors are the format PyTorch expects for all operations, from the _inputs_ (images, texts) to the model parameters (weights and biases).
- **Shape Inspection (`size`/`shape`):** The `size()` (or `shape`) attribute allows us to visualize the **shape** or dimensions of the tensor.
  - For a common training case, like a 2D tensor:
    - The **first parameter** (dimension 0) is typically the **batch size**, indicating how many samples are processed simultaneously.
    - The **second parameter** (dimension 1) is the **number of features** or the size of the individual sample.
- **Data Types (`dtype`):** Although PyTorch can often infer the data type, it is possible (and frequently necessary) to explicitly specify the type (e.g., `torch.float32`, `torch.int64`).
  - **Automatic Casting (Type Promotion):** PyTorch can perform **type promotion** (conversion or _casting_) when performing operations between tensors of different data types, opting for the most precise type.
- **Related Concepts:**
  - **Reshaping:** Involves **changing the shape** of a tensor (its dimensions) while preserving the total number of elements. This is done with functions like `view()` or `reshape()`. It's crucial, for example, for flattening an image before passing it to a linear layer.
  - **Indexing:** Allows for **accessing or modifying** elements or subsets of a tensor using their indices, similar to how it's done with Python arrays or lists (including _slicing_).
