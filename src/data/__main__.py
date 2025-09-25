import argparse

from .commands import compress, fetch, inspect

parser = argparse.ArgumentParser(
    description="Data processing tasks for Cell-Model",
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog="""
Examples:
  python -m src.data fetch                    # Download all datasets
  python -m src.data compress                 # Compress all datasets
  python -m src.data all                      # Fetch and compress all datasets
  python -m src.data inspect                  # Inspect dataset information
  python -m src.data fetch --config custom.toml  # Use custom config file
""",
)

parser.add_argument("command", choices=["fetch", "compress", "all", "inspect"], help="Command to execute")
parser.add_argument("--config", "-c", help="Path to configuration file", default=None)

args = parser.parse_args()

match args.command:
    case "fetch":
        fetch.main(args.config)
    case "compress":
        compress.main(args.config)
    case "all":
        fetch.main(args.config)
        compress.main(args.config)
        inspect.main(args.config)
    case "inspect":
        inspect.main(args.config)
