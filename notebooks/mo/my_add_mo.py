import marimo

__generated_with = "0.14.10"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


# %% [markdown]
# # Marimo Example: Decorator for Long-Running Cells
#
# This notebook demonstrates a decorator that lets you control the execution of a cell with a button, and displays the elapsed time.
#
# The example simply adds two numbers after a short delay.

# %%


def long_running_cell_mo(label: str = "Run", done_message: str = "Done!"):
    """
    Decorator for Marimo cells to control execution with a button and show elapsed time.
    """

    def decorator(func):
        @wraps(func)
        def wrapper(mo, *args, **kwargs):
            run_button = mo.ui.run_button(label=label)
            mo.stop(
                not run_button.value, mo.md(f"Click 👆 **{label}** to run this cell")
            )
            start_time = time.time()
            result = func(mo, *args, **kwargs)
            elapsed = time.time() - start_time
            mo.md(f"{done_message} (Elapsed time: {elapsed:.2f} seconds)")
            return result

        return wrapper

    return decorator


# %% [markdown]
# ## Add Two Numbers (with Button and Timer)
#
# Click the button to execute the addition.


# %%
@app.cell
@long_running_cell_mo(label="Add Numbers", done_message="Addition complete!")
def _(mo):
    """
    Adds two numbers with a simulated delay, and displays the result.
    """
    a = 5
    b = 7
    time.sleep(2)  # Simulate a long-running operation
    result = a + b
    mo.md(f"**Result:** {a} + {b} = {result}")
    return result


# %% [markdown]
# ---
#
# You can edit the numbers in the cell, or reuse the decorator for any other long-running computation.
