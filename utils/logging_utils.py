from dataclasses import dataclass
from typing import Union
from rich.style import Style
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
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
    
    
def display_prediction(task, batch_idx, input_text, prediction, audio_path=None):
    console = Console(width=100)  # Set a reasonable console width
    
    # Create a table for the results
    table = Table(show_header=False, box=None, width=98, padding=(0, 2))  # Add horizontal padding
    table.add_column("Label", style="cyan", width=12, no_wrap=True)  # Fixed width for labels
    table.add_column("Value", style="green", ratio=1, overflow="fold")  # Use remaining space, wrap text
    
    # Add rows with potential long text
    table.add_row("Task", task)
    table.add_row("Batch Index", str(batch_idx))
    # Add empty row for spacing
    table.add_row("", "")
    table.add_row("Input", str(input_text))
    # Add empty row for spacing
    table.add_row("", "")
    table.add_row("Prediction", str(prediction))
    if audio_path:
        # Add empty row for spacing
        table.add_row("", "")
        table.add_row("Audio Output", audio_path)
    
    # Create a panel containing the table
    panel = Panel(
        table,
        title="[bold blue]Prediction Results[/bold blue]",
        border_style="blue",
        padding=(1, 2),
        width=100
    )
    
    console.print("\n")  # Add some spacing
    console.print(panel)
    console.print("\n")  # Add some spacing