import marimo

__generated_with = "0.14.10"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    import time
    return (time,)


@app.cell
def _(time):
    def slow_calc_eg():
        time.sleep(2)
        return 42
    return (slow_calc_eg,)


@app.cell
def _(mo):
    def run_slow_calc(func, label="Click to run calculation"):
        return mo.accordion({
        label: mo.lazy(func, show_loading_indicator=False)
    })
    return (run_slow_calc,)


@app.cell
def _(run_slow_calc, slow_calc_eg):
    run_slow_calc(slow_calc_eg)
    return


@app.cell
def _(mo, time):
    def run_slow_calc_timed(func, label="**Click to run calculation**", done_message="**Calculation completed!**"):
        """
        Wraps a function in a Marimo accordion with lazy evaluation.
        After execution, displays a done message and elapsed time.

        Args:
            func: The function to run.
            label: Label for the accordion/button.
            done_message: Message to show after completion.

        Returns:
            A Marimo accordion UI element.
        """
        def wrapped_func(*args, **kwargs):
            start = time.time()
            result = func(*args, **kwargs)
            elapsed = time.time() - start
            return mo.md(
        f"""
    |   |   |
    |---|---|
    | *Status*      |  {done_message}     |
    | *Elapsed time*| `{elapsed:.2f}` sec |
    | *Result*      | `{result}`          |
        """
    )
    
        return mo.accordion({
            label: mo.lazy(wrapped_func, show_loading_indicator=True)
        })

    return (run_slow_calc_timed,)


@app.cell
def _(run_slow_calc_timed, slow_calc_eg):
    run_slow_calc_timed(slow_calc_eg)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
