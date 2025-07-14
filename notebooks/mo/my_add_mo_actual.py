import marimo

__generated_with = "0.14.10"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    import time
    from functools import wraps
    return mo, time


@app.cell
def _(func, mo, time):
    def long_running_cell_mo(label: str = "Run", done_message: str = "Done!"):
        """
        Decorator for Marimo cells to control execution with a button and show elapsed time.
        """
        def wrapper(*args, **kwargs):
            run_button = mo.ui.run_button(label=label)
            mo.stop(
                not run_button.value,
                mo.md(f"Click 👆 **{label}** to run this cell"),
            )
            start_time = time.time()
            result = func(mo, *args, **kwargs)
            elapsed = time.time() - start_time
            mo.md(f"{done_message} (Elapsed time: {elapsed:.2f} seconds)")
            return result
        return wrapper
    return (long_running_cell_mo,)


@app.cell
def _(long_running_cell_mo, time):
    @long_running_cell_mo(label="Test1", done_message="DONE")
    def my_add(a=5, b=7):
        """
        Adds two numbers with a simulated delay, and displays the result.
        """
        time.sleep(2)  # Simulate a long-running operation
        result = a + b
        return result
    return (my_add,)


@app.cell
def _(my_add):
    my_add()
    return


@app.function
def my_decorator(func):
    def wrapper(*args, **kwargs):
        print("Before function call")
        result = func(*args, **kwargs)
        print("After function call")
        return result
    return wrapper


@app.function
@my_decorator
def greet(name):
    print(f"Hello, {name}!")


@app.cell
def _():
    greet("Bob")
    return


@app.cell
def _():
    from dataclasses import dataclass

    @dataclass
    class Person:
        name: str
        age: int
        city: str = "Sydney"  # Default value

    return (Person,)


@app.cell
def _(Person):
    # Usage example:
    alice = Person(name="Alice", age=30)
    bob = Person(name="Bob", age=25, city="Melbourne")

    print(alice)  # Person(name='Alice', age=30, city='Sydney')
    print(bob)    # Person(name='Bob', age=25, city='Melbourne')
    print(alice.name)  # Alice
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
