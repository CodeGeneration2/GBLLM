# Enable postponed evaluation of type annotations (Python 3.7+ feature),
# allowing undefined classes to be used in type hints.
from __future__ import annotations

import argparse
import importlib
import sys
from types import ModuleType


"""
"""


# #####################################################################################################################
# Define all commands supported by the CLI.
COMMAND_CHOICES = [
    "pairs-to-traces",   # Convert code pairs into execution traces.
    "extract-skills",    # Extract skills/features.
    "build-registry",    # Build the registry.
    "infer",             # Perform inference.
    "reproduce-rq1",     # Reproduce Research Question 1 (RQ1) from the paper.
    "reproduce-rq4",     # Reproduce Research Question 4 (RQ4) from the paper.
    "reproduce-paper",   # Fully reproduce the paper results, including RQ1 and RQ4.
]


# #####################################################################################################################
def _dispatch(module: ModuleType, argv: list[str]) -> None:
    """
    Command dispatcher.

    Temporarily modifies sys.argv, transfers control to the main() function
    of the specified submodule, and restores sys.argv after execution.

    :param module: The submodule to execute.
    :param argv: The list of command-line arguments passed to the submodule.
    """
    # Back up the original sys.argv.
    previous_argv = sys.argv[:]
    try:
        # Pretend sys.argv has the form [module_name, arg1, arg2, ...],
        # so that argparse in the submodule can parse arguments correctly.
        sys.argv = [module.__name__] + argv

        # Call the main entry function of the submodule.
        module.main()
    finally:
        # Restore the original sys.argv regardless of whether execution succeeds.
        sys.argv = previous_argv


# #####################################################################################################################
def _import_module(name: str) -> ModuleType:
    """
    Dynamically import the specified submodule under the current package.

    :param name: The relative name of the submodule, without a leading dot.
    :return: The imported module object.
    :raises SystemExit: If an optional dependency is missing, print a friendly
        installation message and exit the program.
    """
    try:
        # Import using a relative path, equivalent to importing from the current package.
        return importlib.import_module(f".{name}", __package__)
    except ModuleNotFoundError as exc:
        # Catch the module-not-found exception and inspect the missing module name.
        missing_name = getattr(exc, "name", "") or ""

        # If a core module is missing, re-raise the exception directly.
        if missing_name.startswith("effiskill_artifact"):
            raise

        # If a third-party dependency is missing, prompt the user to install the full extras.
        raise SystemExit(
            "Missing optional dependency for this command. "
            'Install the full pipeline extras with `python -m pip install -e ".[full]"` '
            f"and retry. Missing module: {missing_name}"
        ) from exc


# #####################################################################################################################
def _load_command_module(command: str, language: str) -> ModuleType:
    """
    Map and load the corresponding Python execution module based on the
    user-provided command and target language.

    :param command: The command name entered by the user.
    :param language: The target programming language, either "python" or "cpp".
    :return: The corresponding module object.
    :raises RuntimeError: If the command-language combination is unsupported.
    """
    # Define the mapping from (command, language) to module file name.
    module_name = {
        ("pairs-to-traces", "python"): "pairs_to_traces",
        ("pairs-to-traces", "cpp"): "pairs_to_traces_cpp",
        ("extract-skills", "python"): "extract_skills",
        ("extract-skills", "cpp"): "extract_skills_cpp",
        ("build-registry", "python"): "build_registry",
        ("build-registry", "cpp"): "build_registry_cpp",
        ("infer", "python"): "infer_noexec",
        ("infer", "cpp"): "infer_noexec_cpp",
        ("reproduce-rq1", "python"): "paper_rq1",
        ("reproduce-rq4", "python"): "paper_rq4",
    }.get((command, language))

    # If no mapping is found, the command-language combination is invalid.
    if module_name is None:
        raise RuntimeError(f"Unsupported command/language combination: {command} / {language}")

    # Dynamically import and return the module.
    return _import_module(module_name)


# #####################################################################################################################
def build_parser() -> argparse.ArgumentParser:
    """
    Build and return the basic command-line argument parser.

    :return: The ArgumentParser object.
    """
    parser = argparse.ArgumentParser(
        prog="effiskill-artifact",
        description="EffiSkill ASE 2026 artifact CLI.",  # Short description of the CLI.
        add_help=False,  # Disable the default -h/--help option for later custom handling.
    )

    # Receive the main positional command argument. nargs="?" means it is optional,
    # allowing missing commands to be handled manually.
    parser.add_argument("command", nargs="?")

    # Receive the target language argument. Only "python" and "cpp" are allowed.
    # The default language is "python".
    parser.add_argument("--language", choices=["python", "cpp"], default="python")

    # Custom help argument.
    parser.add_argument("-h", "--help", action="store_true")
    return parser


# #####################################################################################################################
def main(argv: list[str] | None = None) -> None:
    """
    Main program entry point.

    It parses the initial arguments and dispatches the remaining arguments
    to the corresponding subcommand module.

    :param argv: The command-line argument list. By default, sys.argv is used.
    """
    parser = build_parser()

    # parse_known_args parses recognized arguments, such as command and --language,
    # and stores unrecognized arguments in the remainder list so they can be
    # passed to the submodule for further parsing.
    args, remainder = parser.parse_known_args(argv)

    # If remainder starts with "--", remove it. This is commonly used to separate
    # main command arguments from subcommand arguments.
    if remainder and remainder[0] == "--":
        remainder = remainder[1:]

    # If no command is provided.
    if args.command is None:
        parser.print_help()  # Print help information.
        if args.help:        # If the user entered -h, exit normally.
            return
        raise SystemExit("Missing command.")  # Otherwise, exit with an error.

    # Validate whether the command is in the supported command list.
    if args.command not in COMMAND_CHOICES:
        parser.error(
            f"argument command: invalid choice: {args.command!r} "
            f"(choose from {', '.join(COMMAND_CHOICES)})"
        )

    # If the --help flag is provided, add it back to the arguments passed to the
    # submodule so that the submodule can print its own help information.
    if args.help:
        remainder = ["--help"] + remainder

    # Load different modules and dispatch based on different commands.

    if args.command == "pairs-to-traces":
        module = _load_command_module(args.command, args.language)
        _dispatch(module, remainder)
        return

    if args.command == "extract-skills":
        module = _load_command_module(args.command, args.language)
        _dispatch(module, remainder)
        return

    if args.command == "build-registry":
        module = _load_command_module(args.command, args.language)
        _dispatch(module, remainder)
        return

    if args.command == "infer":
        module = _load_command_module(args.command, args.language)
        _dispatch(module, remainder)
        return

    # Reproduce RQ1 using the Python module only.
    if args.command == "reproduce-rq1":
        _dispatch(_load_command_module(args.command, "python"), remainder)
        return

    # Reproduce RQ4 using the Python module only.
    if args.command == "reproduce-rq4":
        _dispatch(_load_command_module(args.command, "python"), remainder)
        return

    # Reproduce the full paper in one command by calling the main functions
    # of RQ1 and RQ4 sequentially.
    if args.command == "reproduce-paper":
        _load_command_module("reproduce-rq1", "python").main(remainder)
        _load_command_module("reproduce-rq4", "python").main(remainder)
        return

    # Fallback protection for cases that should theoretically never occur,
    # unless a new command is added without corresponding handling logic.
    raise RuntimeError(f"Unsupported command: {args.command}")


# #####################################################################################################################
# Call main() when the script is run directly.
if __name__ == "__main__":
    main()