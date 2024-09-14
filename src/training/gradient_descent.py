import jax
import jax.numpy as jnp
from jax import grad, jit
import optax
from functools import partial

def stochastic_grad_descent(data, labels, model_fn, optimizer, epochs=100, batch_size=32):
    """
    Trains a model using JAX.

    Args:
        data (jax.numpy.ndarray): Input data of shape (num_samples, num_features).
        labels (jax.numpy.ndarray): Labels of shape (num_samples, ...).
        model_fn (callable): A model function that takes parameters and inputs and returns predictions.
        optimizer (optax.GradientTransformation): An Optax optimizer.
        epochs (int): Number of training epochs.
        batch_size (int): Size of each training batch.

    Returns:
        params: Trained model parameters.
    """
    num_samples = data.shape[0]
    num_batches = num_samples // batch_size

    # Initialize model parameters
    key = jax.random.PRNGKey(0)
    params = model_fn.init(key, data[:batch_size])

    # Initialize optimizer state
    opt_state = optimizer.init(params)

    # Define the loss function
    def loss_fn(params, x, y):
        preds = model_fn.apply(params, x)
        loss = jnp.mean((preds - y) ** 2)  # Mean Squared Error
        return loss

    # Compute gradients and update parameters
    @jit
    def update(params, opt_state, x, y):
        loss, grads = jax.value_and_grad(loss_fn)(params, x, y)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    # Training loop
    for epoch in range(epochs):
        epoch_loss = 0.0
        for i in range(num_batches):
            batch_start = i * batch_size
            batch_end = batch_start + batch_size
            x_batch = data[batch_start:batch_end]
            y_batch = labels[batch_start:batch_end]

            params, opt_state, loss = update(params, opt_state, x_batch, y_batch)
            epoch_loss += loss

        epoch_loss /= num_batches
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch + 1}, Loss: {epoch_loss}")

    return params


