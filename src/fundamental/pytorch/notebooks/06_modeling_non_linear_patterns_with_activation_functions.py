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
    # Modeling Non-Linear Patterns with Activation Functions
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    - Prepare the combined bike and car delivery data, this time applying a technique called **normalization** to help your model train more effectively.
    - Build a non-linear neural network using the **ReLU** activation function.
    - Train your new model to learn the complex, curved relationship in the data.
    - Predict delivery times using your new model and see if it can finally succeed where the linear one failed.
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

    from utils.helper_utils import (
        plot_data,
        plot_final_fit,
        plot_training_progress,
    )
    return nn, optim, plot_data, plot_final_fit, torch


@app.cell
def _(mo):
    mo.md(r"""
    ## Preparing the Non-Linear Data
    """)
    return


@app.cell
def _(torch):
    # Combined dataset: bikes for short distances, cars for longer ones
    distances = torch.tensor(
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
    times = torch.tensor(
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
    return distances, times


@app.cell
def _(distances, plot_data, times):
    # Plot data
    plot_data(distances, times)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## A New Step: Normalizing the Data
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    Normalization is a standard technique that makes the training process more stable and effective by adjusting the scale of the data. This adjustment helps prevent large distance values from dominating the learning process and keeps gradients stable during training.

    - Calculate the mean and standard deviation for the `distances` and `times` tensors.
    - Apply standardization to each tensor using its respective mean and standard deviation, which creates new normalized tensors named `distances_norm` and `times_norm`.
    """)
    return


@app.cell
def _(distances, times):
    # Calculate the mean and standard deviation for the `distances` tensor
    distances_mean = distances.mean()
    distances_std = distances.std()

    # Calculate the mean and standard deviation for the `times` tensor
    times_mean = times.mean()
    times_std = times.std()

    # Apply standardization to the distances
    distances_norm = (distances - distances_mean) / distances_std

    # Apply standardization to the times
    times_norm = (times - times_mean) / times_std
    return (
        distances_mean,
        distances_norm,
        distances_std,
        times_mean,
        times_norm,
        times_std,
    )


@app.cell
def _(distances_norm, plot_data, times_norm):
    # Plot scaled data
    plot_data(distances_norm, times_norm, normalize=True)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Building the Non-Linear Model
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    - `nn.Linear(1, 3)`: This is your first hidden layer. It consists of three neurons each receiving one input feature (the normalized distance). This layer transforms the single input value into three separate values.
    - `nn.ReLU()`: Applies the ReLU activation function to the output of each of the three neurons from the hidden layer. This is the crucial non-linear step that allows your model to create "bends" and learn curves instead of just straight lines.
    - `nn.Linear(3, 1)`: This is your output layer. It takes the three activated values from the previous step as its input and combines them to produce a single final output, which is your predicted (normalized) delivery time.
    """)
    return


@app.cell
def _(nn, torch):
    # Ensures that your results are reproducible and consistent every time
    torch.manual_seed(27)

    model = nn.Sequential(nn.Linear(1, 3), nn.ReLU(), nn.Linear(3, 1))
    return (model,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Training
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
    With the model and training tools ready, it's time to begin the training process.
    """)
    return


@app.cell
def _(distances_norm, loss_function, model, optimizer, times_norm):
    # Training loop
    for epoch in range(3000):
        # Reset the optimizer's gradients
        optimizer.zero_grad()

        # Make predictions (forward pass)
        outputs = model(distances_norm)

        # Calculate the loss
        loss = loss_function(outputs, times_norm)

        # Calculate adjustments (backward pass)
        loss.backward()

        # Update the model's parameters
        optimizer.step()

        # Create a live plot every 50 epochs
        # if (epoch + 1) % 50 == 0:
        #     plot_training_progress(
        #         epoch=epoch, loss=loss, model=model, distances_norm=distances_norm, times_norm=times_norm
        #     )

    print("\nTraining Complete.")
    print(f"\nFinal Loss: {loss.item()}")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Checking the Final Fit
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    Visualize your model's final predicted curve against the original data points
    """)
    return


@app.cell
def _(
    distances,
    distances_norm,
    model,
    plot_final_fit,
    times,
    times_mean,
    times_std,
):
    plot_final_fit(model, distances, times, distances_norm, times_std, times_mean)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Making a Prediction
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    - Take the new input distance and **normalize** it using the same mean and standard deviation from our training data. This step is CRITICAL: your model has no idea about the original scales, it only understands the normalized scaled it was trained on.
    - After the model provides its prediction, you must **de-normalize** the output. This converts the prediction from its normalized scaled back into an understandable value in minutes.
    """)
    return


@app.cell
def _(distances_mean, distances_std, model, times_mean, times_std, torch):
    distance_to_predict = 5.1

    # Use the torch.no_grad() context manager for efficient prediction
    with torch.no_grad():
        # Normalize the input distance
        distance_tensor = torch.tensor(
            [[distance_to_predict]], dtype=torch.float32
        )
        new_distance_norm = (distance_tensor - distances_mean) / distances_std

        # Get the normalized prediction from the model
        predicted_time_norm = model(new_distance_norm)

        # De-normalize the output to get the actual time in minutes
        predicted_time_actual = (predicted_time_norm * times_std) + times_mean

        # --- Decision Making Logic ---
        print(
            f"Prediction for a {distance_to_predict}-mile delivery: {predicted_time_actual.item():.1f} minutes"
        )

        # First, check if the delivery is possible within the 45-minute timeframe
        if predicted_time_actual.item() > 45:
            print("\nDecision: Do NOT promise the delivery in under 45 minutes.")
        else:
            # If it is possible, then determine the vehicle based on the distance
            if distance_to_predict <= 3:
                print(
                    f"\nDecision: Yes, delivery is possible. Since the distance is {distance_to_predict} miles (<= 3 miles), use a bike."
                )
            else:
                print(
                    f"\nDecision: Yes, delivery is possible. Since the distance is {distance_to_predict} miles (> 3 miles), use a car."
                )
    return


if __name__ == "__main__":
    app.run()
