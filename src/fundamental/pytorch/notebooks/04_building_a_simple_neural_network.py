import marimo

__generated_with = "0.17.6"
app = marimo.App(width="columns")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    mo.md(r"""
    # Building a Simple Neural Network
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    Following the Machine Learning (ML) pipeline, you will:

    - **Prepare** delivery data, the distances and times from past orders.
    - **Build** a simple neural network using PyTorch (just one neuron).
    - **Train** it to find the relationship between distance and delivery time.
    - **Predict** whether you can make that 7-mile delivery in time.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Imports
    """)
    return


@app.cell
def _():
    import torch
    import torch.nn as nn
    import torch.optim as optim

    from utils.helper_utils import plot_results, plot_nonlinear_comparison

    # Ensures that results are reproducible and consistent every time
    torch.manual_seed(42)
    return nn, optim, plot_nonlinear_comparison, plot_results, torch


@app.cell
def _(mo):
    mo.md(r"""
    ## The ML Pipeline in Action
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Data Ingestion & Data Preparation
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    - The `distances` tensor contains how far you biked for four recent deliveries (in miles).
    - The `times` tensor shows how long each delivery took (in minutes).
    - `dtype=torch.float32` sets your data type to 32-bit floating point values for precise calculations.
    """)
    return


@app.cell
def _(torch):
    # Distances in miles for recent bike deliveries
    distances = torch.tensor([[1.0], [2.0], [3.0], [4.0]], dtype=torch.float32)

    # Corresponding delivery times in minutes
    times = torch.tensor([[6.96], [12.11], [16.77], [22.21]], dtype=torch.float32)
    return distances, times


@app.cell
def _(mo):
    mo.md(r"""
    ### Model Building
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    A single neuron with one input implements a linear equation:

    $$ Time = W × Distance + B $$

    Your job is to find the best values for the `weight (W)` and `bias (B)` that fit your delivery data.

    - `nn.Sequential()` creates a linear model.
    - `nn.Linear(1, 1)`: The first `1` means it takes one input _(distance)_, and the second `1` means one neuron that is producing one output _(predicted time)_.

    This single linear layer will automatically manage the weight and bias parameters.
    """)
    return


@app.cell
def _(nn):
    # Create a model with one input (distance) and one output (time)
    model = nn.Sequential(nn.Linear(1, 1))
    return (model,)


@app.cell
def _(mo):
    mo.md(r"""
    ### Training
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    **Loss Function**
    - `nn.MSELoss` defines the Mean Squared Error loss function
    - It measures how wrong your predictions are. If you predict 25 minutes but the actual delivery took 30 minutes, the loss function quantifies that 5-minutes error. The model's goal is to minimize this error.

    **Optimizer**
    - `optim.SGC` sets up the Stochastic Gradient Descent optimizer. It adjusts your model's weight and bias parameters based on the errors.
    - `lr=0.0` is the Learning Rate and controls how big each adjustment step is. Too large and you might overshoot the best values; too small and training takes forever.
    """)
    return


@app.cell
def _(model, nn, optim):
    # Define the loss function and optimizer
    loss_function = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    return loss_function, optimizer


@app.cell
def _(mo):
    mo.md(r"""
    The training loop is where your model cycles through the data repeatedly, gradually discovering the relationship between distance and delivery time. You'll train for `500` epochs (complete passes through your data). During each epoch, these steps occurs:
    - `optimizer.zero_grad()`: Cleans gradients from the previous round. Without this, PyTorch accumulate adjustments, which could break the learning process.
    - `outputs = model(distances)`: Performs the "forward pass", where the model makes predictions based on the input `distances`.
    - `loss = loss_function(outputs, times)`: Calculates how wrong the predicted `outputs` are by comparing them to the actual delivery `times`.
    - `loss.backward()`: The "backward pass" (backpropagation) is performed, which calculates exactly how to adjust the weight and bias to reduce the error.
    - `optimizer.step()`: Updates the model's parameters using those calculated adjustments.
    """)
    return


@app.cell
def _(distances, loss_function, model, optimizer, times):
    # Training loop
    for epoch in range(500):
        # Reset the optimizer's gradients
        optimizer.zero_grad()
        # Make predictions (forward pass)
        outputs = model(distances)
        # Calculate the loss
        loss = loss_function(outputs, times)
        # Calculate adjustments (backward pass)
        loss.backward()
        # Update the model's parameters
        optimizer.step()
        # Print loss every 50 epochs
        if (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch + 1}: Loss = {loss.item()}")
    return


@app.cell
def _(mo):
    mo.md(r"""
    **Visualizing the Training Results**
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    Let's plot the model's predictions as a line against the actual delivery data points. The helper function, `plot_results`, will show:
    - The original data points (actual deliveries).
    - The line the model learned (its predictions).
    - How well they match.
    """)
    return


@app.cell
def _(distances, model, plot_results, times):
    plot_results(model, distances, times)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Evaluation
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    #### Make Your Prediction
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    Let's make a data-driven prediction for a specific delivery:

    - Create the `distance_to_predict` variable
    - The entire prediction process is wrapped in a `with torch.no_grad()` block.
      - This tella PyTorch you're not training anymore, just making a prediction. This makes the process faster and more efficient.
    - Create a input tensor that must be formatted as a 2D tensor `[[7.0]]`.
    - The trained model is called with this new tensor to generate a `predicted_time`.
    - After getting the prediction (tensor), the code extracts the actual numerical value from it using `.item()`.
    """)
    return


@app.cell
def _(model, torch):
    distance_to_predict = 7.0

    # Use the torch.no_grad() context manager for efficient predictions
    with torch.no_grad():
        # Convert the Python variable into a 2D PyTorch tensor that the model expects
        new_distance = torch.tensor([[distance_to_predict]], dtype=torch.float32)

        # Pass the new data to the trained model to get a prediction
        predicted_time = model(new_distance)

        # Use .item() to extract the scalar value from the tensor for printing
        print(
            f"Prediction for a {distance_to_predict}-mile delivery: {predicted_time.item():.1f} minutes"
        )

        # Use the scalar value in a conditional statement to make the final decision
        if predicted_time.item() > 30:
            print("\nDecision: Do NOT take the job. You will likely be late.")
        else:
            print("\nDecision: Take the job. You can make it!")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Inspecting the Model's Learning
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    Now that you have a working model, let's see the exact relationship it leraned from the data. You can do this by inspecting the model's internal parameters, the final **weight** and **bias** values it discovered during training. These values define the precise line your model is now using to make predictions.
    """)
    return


@app.cell
def _(model):
    # Access the first (and only) layer in the sequential model
    layer = model[0]

    # Get weights and bias
    weights = layer.weight.data.numpy()
    bias = layer.bias.data.numpy()

    print(f"Weight: {weights}")
    print(f"Bias: {bias}")
    return


@app.cell
def _(mo):
    mo.md(r"""
    - **Weight:** This means that for each additional mile, the model predicts the delivery time will increase by about 5.0 minutes.
    - **Bias:** This represents the base time for any delivery, regardless of distance. Think of it as the time needed to pick up the order and get on your bike.

    The model now can predict delivery times for any distance using the equation:

    $$Time = 5.0 * Distance + 2.0$$
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Testing Your Model on More Complex Data
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    Define a new dataset, which includes the original bike data plus new data points for longer-distance car deliveries.
    """)
    return


@app.cell
def _(torch):
    # Combined dataset: bikes for short distances, cars for longer ones
    new_distances = torch.tensor(
        [
            [1.0],
            [1.5],
            [2.0],
            [2.5],
            [3.0],
            [3.5],
            [4.0],
            [4.5],
            [5.0],
            [5.5],
            [6.0],
            [6.5],
            [7.0],
            [7.5],
            [8.0],
            [8.5],
            [9.0],
            [9.5],
            [10.0],
            [10.5],
            [11.0],
            [11.5],
            [12.0],
            [12.5],
            [13.0],
            [13.5],
            [14.0],
            [14.5],
            [15.0],
            [15.5],
            [16.0],
            [16.5],
            [17.0],
            [17.5],
            [18.0],
            [18.5],
            [19.0],
            [19.5],
            [20.0],
        ],
        dtype=torch.float32,
    )

    # Corresponding delivery times in minutes
    new_times = torch.tensor(
        [
            [6.96],
            [9.67],
            [12.11],
            [14.56],
            [16.77],
            [21.7],
            [26.52],
            [32.47],
            [37.15],
            [42.35],
            [46.1],
            [52.98],
            [57.76],
            [61.29],
            [66.15],
            [67.63],
            [69.45],
            [71.57],
            [72.8],
            [73.88],
            [76.34],
            [76.38],
            [78.34],
            [80.07],
            [81.86],
            [84.45],
            [83.98],
            [86.55],
            [88.33],
            [86.83],
            [89.24],
            [88.11],
            [88.16],
            [91.77],
            [92.27],
            [92.13],
            [90.73],
            [90.39],
            [92.98],
        ],
        dtype=torch.float32,
    )
    return new_distances, new_times


@app.cell
def _(mo):
    mo.md(r"""
    Use the trained `model` to generates predictions on the `new_distances`.
    """)
    return


@app.cell
def _(model, new_distances, torch):
    # Use the already-trained linear model to make predictions
    with torch.no_grad():
        predictions = model(new_distances)
    return (predictions,)


@app.cell
def _(mo):
    mo.md(r"""
    Calculate the `new_loss` between the model's predictions and the actual times.
    """)
    return


@app.cell
def _(loss_function, new_times, predictions):
    # Calculate the new loss
    new_loss = loss_function(predictions, new_times)
    print(f"Loss on new, combined data: {new_loss.item():.2f}")
    return


@app.cell
def _(mo):
    mo.md(r"""
    The loss is much higher and a quick look at the visualization showed why:

    - **Actual Data (orange points):** Delivery times follow a more complex pattern. Bikes take short, direct routes. Cars deal with city traffic, then speed up on highways. The relationship between distance and time isn’t smooth or consistent, it bends and shifts.

    - **Model Predictions (green line):** Your model can only draw a straight line. It learned one pattern: each mile adds about the same amount of time. But now that assumption no longer holds.

    To capture these non-linear patterns, you need to give your model the ability to learn curves, not just lines.
    """)
    return


@app.cell
def _(model, new_distances, new_times, plot_nonlinear_comparison):
    plot_nonlinear_comparison(model, new_distances, new_times)
    return


if __name__ == "__main__":
    app.run()
