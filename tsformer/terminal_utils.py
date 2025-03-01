import os
import time
from datetime import datetime
from typing import Dict, List, Any, Optional, Union

# ANSI color codes for terminal output
class Colors:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"
    
    # Foreground colors
    BLACK = "\033[30m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"
    
    # Bright foreground colors
    BRIGHT_BLACK = "\033[90m"
    BRIGHT_RED = "\033[91m"
    BRIGHT_GREEN = "\033[92m"
    BRIGHT_YELLOW = "\033[93m"
    BRIGHT_BLUE = "\033[94m"
    BRIGHT_MAGENTA = "\033[95m"
    BRIGHT_CYAN = "\033[96m"
    BRIGHT_WHITE = "\033[97m"
    
    # Background colors
    BG_BLACK = "\033[40m"
    BG_RED = "\033[41m"
    BG_GREEN = "\033[42m"
    BG_YELLOW = "\033[43m"
    BG_BLUE = "\033[44m"
    BG_MAGENTA = "\033[45m"
    BG_CYAN = "\033[46m"
    BG_WHITE = "\033[47m"


def get_terminal_width() -> int:
    """Get the width of the terminal."""
    try:
        return os.get_terminal_size().columns
    except (AttributeError, OSError):
        return 80


def print_header(text: str, style: str = "double") -> None:
    """Print a styled header with the given text."""
    width = get_terminal_width()
    text = f" {text} "
    
    if style == "double":
        border_char = "="
    elif style == "single":
        border_char = "-"
    elif style == "hash":
        border_char = "#"
    else:
        border_char = "="
    
    side_width = (width - len(text)) // 2
    
    header = (f"{Colors.BOLD}{Colors.BRIGHT_CYAN}"
              f"{border_char * side_width}{text}{border_char * (side_width + (width - len(text)) % 2)}"
              f"{Colors.RESET}")
    
    print("\n" + header)


def print_subheader(text: str) -> None:
    """Print a styled subheader with the given text."""
    print(f"\n{Colors.BOLD}{Colors.BRIGHT_BLUE}>> {text}{Colors.RESET}")


def print_success(text: str) -> None:
    """Print a success message."""
    print(f"{Colors.BRIGHT_GREEN}✅ {text}{Colors.RESET}")


def print_error(text: str) -> None:
    """Print an error message."""
    print(f"{Colors.BRIGHT_RED}❌ {text}{Colors.RESET}")


def print_warning(text: str) -> None:
    """Print a warning message."""
    print(f"{Colors.BRIGHT_YELLOW}⚠️ {text}{Colors.RESET}")


def print_info(text: str) -> None:
    """Print an informational message."""
    print(f"{Colors.BRIGHT_CYAN}ℹ️ {text}{Colors.RESET}")


def print_progress(text: str) -> None:
    """Print a progress message."""
    print(f"{Colors.BRIGHT_MAGENTA}→ {text}{Colors.RESET}")


def print_metric(name: str, value: Union[float, int], is_good: bool = True) -> None:
    """Print a metric with appropriate coloring based on whether the value is good or bad."""
    color = Colors.BRIGHT_GREEN if is_good else Colors.BRIGHT_RED
    print(f"  {Colors.BOLD}{name}:{Colors.RESET} {color}{value}{Colors.RESET}")


def print_metrics_table(metrics: Dict[str, Any], title: str = "Metrics") -> None:
    """Print a table of metrics with a title."""
    print_subheader(title)
    
    # Find the longest metric name for alignment
    max_name_length = max(len(name) for name in metrics.keys())
    
    for name, value in metrics.items():
        # Determine if the value is "good" (you might want to customize this logic)
        is_good = True  # Default assumption
        if name.lower() in ["loss", "error", "mae", "mse", "rmse", "mape"]:
            is_good = False  # Lower is better for these metrics
        
        # Format the metric value
        if isinstance(value, float):
            formatted_value = f"{value:.6f}"
        else:
            formatted_value = str(value)
            
        # Add padding for alignment and print
        padded_name = name.ljust(max_name_length)
        color = Colors.BRIGHT_GREEN if is_good else Colors.BRIGHT_RED
        print(f"  {Colors.BOLD}{padded_name}:{Colors.RESET} {color}{formatted_value}{Colors.RESET}")


def print_progress_bar(iteration: int, total: int, prefix: str = '', suffix: str = '', 
                       decimals: int = 1, length: int = 30, fill: str = '█') -> None:
    """
    Print a progress bar to the terminal.
    
    Args:
        iteration: Current iteration (0-based)
        total: Total iterations
        prefix: Prefix string
        suffix: Suffix string
        decimals: Decimal places for percentage
        length: Character length of the bar
        fill: Bar fill character
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    
    print(f'\r{Colors.BRIGHT_BLUE}{prefix}{Colors.RESET} |{Colors.BRIGHT_CYAN}{bar}{Colors.RESET}| {percent}% {suffix}', end='\r')
    
    # Print new line on complete
    if iteration == total:
        print()


def format_time(seconds: float) -> str:
    """Format time in seconds to a human-readable string."""
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = seconds // 60
        seconds %= 60
        return f"{int(minutes)}m {int(seconds)}s"
    else:
        hours = seconds // 3600
        seconds %= 3600
        minutes = seconds // 60
        seconds %= 60
        return f"{int(hours)}h {int(minutes)}m {int(seconds)}s"


def print_time_elapsed(start_time: float, prefix: str = "Time elapsed") -> None:
    """Print the time elapsed since start_time."""
    elapsed = time.time() - start_time
    print(f"{Colors.BRIGHT_BLUE}{prefix}:{Colors.RESET} {Colors.BRIGHT_MAGENTA}{format_time(elapsed)}{Colors.RESET}")


def print_config(config: Dict[str, Any], title: str = "Configuration") -> None:
    """Print a configuration dictionary in a nicely formatted way."""
    print_header(title)
    
    # Find the longest config key for alignment
    max_key_length = max(len(key) for key in config.keys())
    
    # Group by categories if keys have common prefixes
    categories = {}
    standalone_keys = []
    
    for key in config.keys():
        if "_" in key:
            category = key.split("_")[0]
            if category not in categories:
                categories[category] = []
            categories[category].append(key)
        else:
            standalone_keys.append(key)
    
    # Print standalone keys first
    for key in standalone_keys:
        padded_key = key.ljust(max_key_length)
        print(f"  {Colors.BOLD}{padded_key}:{Colors.RESET} {Colors.BRIGHT_YELLOW}{config[key]}{Colors.RESET}")
    
    # Print categorized keys
    for category, keys in categories.items():
        print(f"\n  {Colors.UNDERLINE}{Colors.BRIGHT_BLUE}{category.capitalize()}{Colors.RESET}")
        for key in sorted(keys):
            # Remove the category prefix for cleaner display
            display_key = key.replace(f"{category}_", "")
            padded_key = display_key.ljust(max_key_length - len(category) - 1)
            print(f"    {Colors.BOLD}{padded_key}:{Colors.RESET} {Colors.BRIGHT_YELLOW}{config[key]}{Colors.RESET}")


def print_timestamp(prefix: str = "Current time") -> None:
    """Print the current timestamp."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"{Colors.BRIGHT_BLUE}{prefix}:{Colors.RESET} {Colors.BRIGHT_CYAN}{timestamp}{Colors.RESET}")


def clear_terminal() -> None:
    """Clear the terminal screen."""
    os.system('cls' if os.name == 'nt' else 'clear')


def print_section_separator() -> None:
    """Print a section separator line."""
    width = get_terminal_width()
    print(f"\n{Colors.BRIGHT_BLACK}{'-' * width}{Colors.RESET}\n") 