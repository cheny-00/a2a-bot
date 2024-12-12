from dataclasses import dataclass
from typing import Union
from rich.style import Style
from rich.console import Console
from rich.table import Table

@dataclass
class LightRichProgressBarTheme:
    """Styles to associate to different base components.

    Args:
        description: Style for the progress bar description. For eg., Epoch x, Testing, etc.
        progress_bar: Style for the bar in progress.
        progress_bar_finished: Style for the finished progress bar.
        progress_bar_pulse: Style for the progress bar when `IterableDataset` is being processed.
        batch_progress: Style for the progress tracker (i.e 10/50 batches completed).
        time: Style for the processed time and estimate time remaining.
        processing_speed: Style for the speed of the batches being processed.
        metrics: Style for the metrics

    https://rich.readthedocs.io/en/stable/style.html

    """

    description: Union[str, "Style"] = "grey37"
    progress_bar: Union[str, "Style"] = "#834fcb" 
    progress_bar_finished: Union[str, "Style"] = "#04d5ca"
    progress_bar_pulse: Union[str, "Style"] = "#dd6707"
    batch_progress: Union[str, "Style"] = "grey37"
    time: Union[str, "Style"] = "medium_purple4"
    processing_speed: Union[str, "Style"] = "pale_turquoise4"
    metrics: Union[str, "Style"] = "grey37"
    metrics_text_delimiter: str = " "
    metrics_format: str = ".3f"
    
    

def show_optimizer_details(optimizer_name, optimizer_params):
    console = Console()

    # Create a Rich Table
    table = Table(title="Optimizer Details", show_header=True, header_style="bold magenta")

    # Add columns
    table.add_column("Parameter", justify="right", style="cyan", no_wrap=True)
    table.add_column("Value", justify="left", style="green")

    # Add optimizer name
    table.add_row("Optimizer Name", optimizer_name)

    # Add optimizer parameters
    for param, value in optimizer_params.items():
        table.add_row(param, str(value))

    # Print the table to the console
    console.print(table)