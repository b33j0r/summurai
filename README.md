
# Summarai

Summarai is an AI-enabled directory summarizer and query tool designed to help you manage, summarize, and query files and directories. Whether you're working with textual or non-textual files, Summarai provides powerful tools to generate summaries, check relevance, and clean up metadata files.

## Features
- **Directory Summarization**: Summarizes textual files and generates metadata for quick reference.
- **Querying**: Search for relevant files based on user-defined queries.
- **Interactive Mode**: Explore directories interactively through a command-line interface.
- **Cleanup**: Remove `.summarai.json` files from your project with a simple command.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/b33j0r/summarai.git
   cd summarai
   ```

2. Install the package:
   ```bash
   pip install .
   ```

   Alternatively, you can build and install the package locally:
   ```bash
   pip install hatch
   hatch build
   pip install dist/summarai-0.1.0-py3-none-any.whl
   ```

3. Verify installation:
   ```bash
   summarai --help
   ```

## Usage

Summarai provides the following commands:

### Indexing a Directory
```bash
summarai index [path] [--parallel] [--workers <number>] [--debug]
```
Indexes the specified directory and generates summaries.

### Querying Indexed Summaries
```bash
summarai query [path] <query> [--max_depth <number>] [--debug]
```
Queries the indexed summaries for relevant files.

### Interactive Query Mode
```bash
summarai interactive [path] [--max_depth <number>] [--debug]
```
Launches an interactive shell for querying summaries.

### Cleaning Up Metadata Files
```bash
summarai clean [path] [--debug]
```
Removes `.summarai.json` files from the specified directory and its subdirectories.

### Example Commands
Note that the current directory (`.`) is used as the default path if not specified.
It's shown here to show where the path argument should be placed.

```bash
# Index a directory
summarai index .

# Query summaries
summarai query . "example query"

# Interactive mode
summarai interactive .

# Clean up metadata files
summarai clean .
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

Summarai was originally created with [ChatGPT o1-mini](https://openai.com/chatgpt).

---

Feel free to contribute to the project by submitting issues or pull requests on [GitHub](https://github.com/b33j0r/summarai).
