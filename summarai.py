# summarai.py

import json
import logging
from pathlib import Path

from tqdm import tqdm
from queue import Queue
from concurrent.futures import ThreadPoolExecutor
import argparse
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List
import requests

DEFAULT_MODEL = "llama3.1"

# ----------------------------
# Logging Configuration
# ----------------------------

def setup_logging(debug: bool):
    """Sets up logging configuration."""
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.StreamHandler()
        ]
    )

logger = logging.getLogger(__name__)

# ----------------------------
# Prompt Management
# ----------------------------

class PromptManager:
    """Manages prompts based on the current operation."""

    def get_generate_summary_prompt(self, text: str) -> str:
        """Generates a prompt for summarizing text."""
        return (
            "You are a helpful assistant that summarizes files for a file index record. "
            "The user wishes to know what each file is about. \n\n"
            "Do not include other remarks. \n\n"
            f"Please provide a concise summary of the following text:\n\n{text}"
        )

    def get_check_relevance_prompt(self, summary: str, query: str) -> str:
        """Generates a prompt for checking relevance of a summary to a query."""
        return (
            "You determine the relevance of a summary to a query.\n\n"
            f"Determine if the following summary is relevant to the query.\n\n"
            f"Query: {query}\n\nSummary: {summary}\n\n"
            "Answer with 'Yes' or 'No'."
        )

# ----------------------------
# LLM Abstraction Layer
# ----------------------------

class LLMInterface(ABC):
    """Abstract base class for LLM implementations."""

    def __init__(self, prompt_manager: PromptManager):
        self.prompt_manager = prompt_manager

    @abstractmethod
    def generate_text_summary(self, text: str, max_tokens: int = 150) -> str:
        pass

    @abstractmethod
    def check_relevance(self, summary: str, query: str) -> bool:
        pass

# class OpenAI_LLM(LLMInterface):
#     """OpenAI GPT-4 implementation of the LLMInterface."""
#
#     # [OpenAI implementation remains unchanged and commented out]
#

class Ollama_LLM(LLMInterface):
    """Ollama LLM implementation of the LLMInterface."""

    def __init__(self, api_url: str, prompt_manager: PromptManager, debug: bool = False):
        super().__init__(prompt_manager)
        self.api_url = api_url
        self.debug = debug
        self.model = DEFAULT_MODEL

    def _send_request(self, model: str, prompt: str) -> str:
        """Sends a request to the Ollama server."""
        try:
            if self.debug:
                logger.debug(f"Sending request to Ollama model '{model}' with prompt: {prompt}")
            response = requests.post(
                self.api_url + "/api/generate",
                json={"model": model, "prompt": prompt, "stream": False},
                timeout=10
            )
            response.raise_for_status()  # Raise an error for non-200 responses
            result = response.json()
            if "response" in result:
                generated_text = result["response"]
                if self.debug:
                    logger.debug(f"Received response from Ollama: {generated_text}")
                return generated_text.strip()
            else:
                raise ValueError(f"Unexpected response structure: {result}")
        except requests.exceptions.RequestException as e:
            logger.error(f"Error communicating with Ollama server: {e}")
            return "Error: Unable to connect to Ollama server."

    def generate_text_summary(self, text: str, max_tokens: int = 150) -> str:
        """Generates a summary using the Ollama server."""
        prompt = self.prompt_manager.get_generate_summary_prompt(text)
        return self._send_request(model=self.model, prompt=prompt)

    def check_relevance(self, summary: str, query: str) -> bool:
        """Checks the relevance of a summary to a query using the Ollama server."""
        prompt = self.prompt_manager.get_check_relevance_prompt(summary, query)
        relevance_response = self._send_request(model=self.model, prompt=prompt)
        relevance_result = relevance_response.lower().startswith("yes")
        if self.debug:
            logger.debug(f"Relevance result from Ollama: {relevance_result}")
        return relevance_result

class Mock_LLM(LLMInterface):
    """Mock implementation of the LLMInterface for testing purposes."""

    def __init__(self, prompt_manager: PromptManager, debug: bool = False):
        super().__init__(prompt_manager)
        self.debug = debug

    def generate_text_summary(self, text: str, max_tokens: int = 150) -> str:
        summary = f"Mock summary of text with {len(text)} characters."
        if self.debug:
            logger.debug(f"Mock generate_text_summary called. Summary: {summary}")
        return summary

    def check_relevance(self, summary: str, query: str) -> bool:
        # Simple mock logic: return True if query is in summary
        is_relevant = query.lower() in summary.lower()
        if self.debug:
            logger.debug(f"Mock check_relevance called. Summary contains query: {is_relevant}")
        return is_relevant

class Caching_LLM(LLMInterface):
    """Caching wrapper around another LLMInterface implementation."""

    def __init__(self, llm: LLMInterface, cache_file: Path = Path('llm_cache.json'), debug: bool = False):
        super().__init__(llm.prompt_manager)
        self.llm = llm
        self.cache_file = cache_file
        self.debug = debug
        self.cache: Dict[str, Any] = self.load_cache()

    def load_cache(self) -> Dict[str, Any]:
        if self.cache_file.exists():
            try:
                with self.cache_file.open('r', encoding='utf-8') as f:
                    cache = json.load(f)
                if self.debug:
                    logger.debug(f"Loaded cache with {len(cache)} entries.")
                return cache
            except Exception as e:
                logger.error(f"Failed to load cache file {self.cache_file}: {e}")
        return {}

    def save_cache(self):
        try:
            with self.cache_file.open('w', encoding='utf-8') as f:
                json.dump(self.cache, f, indent=4)
            if self.debug:
                logger.debug(f"Saved cache with {len(self.cache)} entries.")
        except Exception as e:
            logger.error(f"Failed to save cache file {self.cache_file}: {e}")

    def generate_text_summary(self, text: str, max_tokens: int = 150) -> str:
        key = f"summary::{hash(text)}"
        if key in self.cache:
            if self.debug:
                logger.debug(f"Cache hit for generate_text_summary with key: {key}")
            return self.cache[key]
        summary = self.llm.generate_text_summary(text, max_tokens)
        self.cache[key] = summary
        self.save_cache()
        return summary

    def check_relevance(self, summary: str, query: str) -> bool:
        key = f"relevance::{hash(summary)}::{hash(query)}"
        if key in self.cache:
            if self.debug:
                logger.debug(f"Cache hit for check_relevance with key: {key}")
            return self.cache[key]
        is_relevant = self.llm.check_relevance(summary, query)
        self.cache[key] = is_relevant
        self.save_cache()
        return is_relevant

class Logging_LLM(LLMInterface):
    """LLM implementation that logs all interactions."""

    def __init__(self, child_llm: LLMInterface, debug: bool = False):
        super().__init__(child_llm.prompt_manager)
        self.child_llm = child_llm
        self.debug = debug

    def generate_text_summary(self, text: str, max_tokens: int = 150) -> str:
        logger.debug(f"Logging_LLM: Generating text summary for text of length {len(text)}")
        summary = self.child_llm.generate_text_summary(text, max_tokens)
        logger.debug(f"Logging_LLM: Received summary: {summary}")
        return summary

    def check_relevance(self, summary: str, query: str) -> bool:
        logger.debug(f"Logging_LLM: Checking relevance for query: '{query}' with summary: '{summary}'")
        is_relevant = self.child_llm.check_relevance(summary, query)
        logger.debug(f"Logging_LLM: Relevance result: {is_relevant}")
        return is_relevant

# ----------------------------
# File Summarization Components
# ----------------------------

TEXTUAL_EXTENSIONS = {
    '.txt', '.md', '.py', '.java', '.c', '.cpp', '.json',
    '.csv', '.html', '.css', '.js', '.rb', '.go', '.rs'
}
NON_TEXTUAL_EXTENSIONS = {
    '.png', '.jpg', '.jpeg', '.gif', '.bmp', '.pdf', '.docx',
    '.xlsx', '.pptx', '.zip', '.rar', '.7z', '.exe', '.dll'
}

IGNORED_PATHS = {
    '.DS_Store',
    '.git',
    '.idea',
    '.ipynb_checkpoints',
    '.summarai.json',
    '.venv',
    '.vscode',
    '__pycache__',
    'node_modules',
    'venv',
}

def is_textual_file(file_path: Path) -> bool:
    return file_path.suffix.lower() in TEXTUAL_EXTENSIONS

def is_non_textual_file(file_path: Path) -> bool:
    return file_path.suffix.lower() in NON_TEXTUAL_EXTENSIONS

def get_files(directory: Path) -> List[Path]:
    return [
        f for f in directory.iterdir()
        if f.is_file() and not any(ignored in f.parts for ignored in IGNORED_PATHS)
    ]

def get_subdirectories(directory: Path) -> List[Path]:
    return [
        d for d in directory.iterdir()
        if d.is_dir() and not any(ignored in d.parts for ignored in IGNORED_PATHS)
    ]

def generate_file_description(file_path: Path) -> str:
    try:
        size_mb = file_path.stat().st_size / (1024 * 1024)
        extension = file_path.suffix.lower()
        description = f"{size_mb:.2f}MB {extension[1:].upper()} file"
        logger.debug(f"Generated description for {file_path}: {description}")
        return description
    except Exception as e:
        logger.error(f"Error generating file description for {file_path}: {e}")
        return "File description unavailable due to an error."

def summarize_directory(directory: Path, llm: LLMInterface):
    summary_data = {}
    entries = []

    # Process files
    files = get_files(directory)
    for file in files:
        assert not any(ignored in file.parts for ignored in IGNORED_PATHS)
        assert file.name != '.summarai.json'

        extension = file.suffix.lower()

        if is_textual_file(file):
            try:
                content = file.read_text(encoding='utf-8')
                file_summary = llm.generate_text_summary(content)
                entries.append(
                    {
                        "type": "file",
                        "name": file.name,
                        "summary": file_summary
                    }
                )
                logger.debug(f"Summarized textual file: {file}")
            except Exception as e:
                logger.error(f"Failed to process textual file {file}: {e}")
                entries.append(
                    {
                        "type": "file",
                        "name": file.name,
                        "summary": "Summary unavailable due to an error."
                    }
                )
        elif is_non_textual_file(file):
            description = generate_file_description(file)
            entries.append(
                {
                    "type": "file",
                    "name": file.name,
                    "summary": description
                }
            )
            logger.debug(f"Described non-textual file: {file}")
        else:
            # Handle other file types if necessary
            description = generate_file_description(file)
            entries.append(
                {
                    "type": "file",
                    "name": file.name,
                    "summary": description
                }
            )
            logger.debug(f"Described unknown file type: {file}")

    # Generate meta_summary based on all textual file summaries
    textual_summaries = "\n".join(
        [entry["summary"] for entry in entries if entry["type"] == "file" and is_textual_file(directory / entry["name"])]
    )
    if textual_summaries:
        meta_summary = llm.generate_text_summary(textual_summaries)
    else:
        meta_summary = "No textual files in this directory."

    summary_data['meta_summary'] = meta_summary

    # Process subdirectories
    subdirs = get_subdirectories(directory)
    for subdir in subdirs:
        assert not any(ignored in subdir.parts for ignored in IGNORED_PATHS)

        subdir_summary_path = subdir / '.summarai.json'
        if subdir_summary_path.exists():
            try:
                sub_summary_data = subdir_summary_path.read_text(encoding='utf-8')
                sub_summary_json = json.loads(sub_summary_data)
                sub_summary = sub_summary_json.get('meta_summary', 'No summary available.')
                entries.append(
                    {
                        "type": "directory",
                        "name": subdir.name,
                        "summary": sub_summary
                    }
                )
                logger.debug(f"Included summary for subdirectory: {subdir}")
            except Exception as e:
                logger.error(f"Failed to read {subdir_summary_path}: {e}")
                entries.append(
                    {
                        "type": "directory",
                        "name": subdir.name,
                        "summary": "Summary unavailable due to an error."
                    }
                )
        else:
            entries.append(
                {
                    "type": "directory",
                    "name": subdir.name,
                    "summary": "No summary available."
                }
            )
            logger.debug(f"No summary found for subdirectory: {subdir}")

    summary_data['entries'] = entries

    # Write to .summarai.json
    summarai_path = directory / '.summarai.json'
    try:
        summarai_path.write_text(json.dumps(summary_data, indent=4), encoding='utf-8')
        logger.info(f"Summarized directory: {directory}")
    except Exception as e:
        logger.error(f"Failed to write {summarai_path}: {e}")

# ----------------------------
# Querying Components
# ----------------------------

def query_summarai(root_directory: Path, query: str, llm: LLMInterface, max_depth: int = 10) -> List[Path]:
    queue = Queue()
    queue.put((root_directory, 0))
    relevant_files = []

    while not queue.empty():
        current_dir, depth = queue.get()
        if depth > max_depth:
            logger.debug(f"Max depth reached at directory: {current_dir}")
            continue

        summarai_path = current_dir / '.summarai.json'
        if not summarai_path.exists():
            logger.warning(f"No .summarai.json found in directory: {current_dir}")
            continue

        try:
            summary_data = json.loads(summarai_path.read_text(encoding='utf-8'))
        except Exception as e:
            logger.error(f"Failed to read {summarai_path}: {e}")
            continue

        # Check meta_summary for relevance
        meta_summary = summary_data.get('meta_summary', '')
        is_relevant = llm.check_relevance(meta_summary, query)
        logger.debug(f"Directory '{current_dir}' relevance: {is_relevant}")

        if is_relevant:
            # Collect relevant files
            for entry in summary_data.get('entries', []):
                if entry["type"] == "file":
                    file_path = current_dir / entry["name"]
                    relevant_files.append(file_path)
                    logger.debug(f"Added relevant file: {file_path}")

        # Enqueue subdirectories for BFS
        for entry in summary_data.get('entries', []):
            if entry["type"] == "directory":
                subdir_path = current_dir / entry["name"]
                queue.put((subdir_path, depth + 1))
                logger.debug(f"Enqueued subdirectory: {subdir_path} at depth {depth + 1}")

    return relevant_files

# ----------------------------
# Main Indexing and Querying Functions
# ----------------------------

def index_directory_tree(root_directory: Path, llm: LLMInterface):
    """Traverses the directory tree bottom-up and summarizes each directory."""
    directories = sorted(root_directory.rglob('*'), key=lambda p: len(p.parts), reverse=True)
    for directory in directories:
        if directory.is_dir() and not any(ignored in directory.parts for ignored in IGNORED_PATHS):
            summarize_directory(directory, llm)
    # Finally, summarize the root directory
    if root_directory.is_dir():
        summarize_directory(root_directory, llm)

def index_directory_tree_parallel(root_directory: Path, llm: LLMInterface, max_workers: int = 4):
    """Traverses the directory tree bottom-up and summarizes each directory in parallel."""
    directories = sorted(root_directory.rglob('*'), key=lambda p: len(p.parts), reverse=True)
    directories = [d for d in directories if d.is_dir() and not any(ignored in d.parts for ignored in IGNORED_PATHS)]
    directories.append(root_directory)  # Ensure root is included

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        list(tqdm(executor.map(lambda d: summarize_directory(d, llm), directories), total=len(directories)))

def interactive_query(root_directory: Path, llm: LLMInterface, max_depth: int = 10):
    """Provides an interactive shell for querying."""
    print("Enter your queries (type 'exit' to quit):")
    while True:
        try:
            query = input("Query> ")
        except (EOFError, KeyboardInterrupt):
            print("\nExiting interactive mode.")
            break

        if query.lower() in {'exit', 'quit'}:
            break
        results = query_summarai(root_directory, query, llm, max_depth=max_depth)
        if results:
            print("\nRelevant files:")
            for file in results:
                print(file)
        else:
            print("\nNo relevant files found.")
        print("\n")

# ----------------------------
# Clean Function
# ----------------------------

def clean_directory_tree(root_directory: Path):
    """Cleans up .summarai.json files from the directory tree."""
    summarai_files = list(root_directory.rglob('.summarai.json'))
    if not summarai_files:
        logger.info("No .summarai.json files found to clean.")
        return

    for file in summarai_files:
        try:
            file.unlink()  # Deletes the file
            logger.info(f"Deleted: {file}")
        except Exception as e:
            logger.error(f"Failed to delete {file}: {e}")

# ----------------------------
# Main Function
# ----------------------------

def main():
    parser = argparse.ArgumentParser(description="Summarai: AI-enabled directory summarizer and query tool.")
    parser.add_argument('--llm', choices=['openai', 'mock', 'ollama'], default='ollama', help='LLM implementation to use.')
    parser.add_argument('--cache', action='store_true', help='Enable caching for LLM interactions.')

    subparsers = parser.add_subparsers(dest='command', help='Commands')

    # Common arguments
    parser.add_argument('--debug', action='store_true', help='Enable debug logging.')

    # Index command
    index_parser = subparsers.add_parser('index', help='Index and summarize the directory tree.')
    index_parser.add_argument('root', nargs='?', default='.', type=Path, help='Root directory to index.')
    index_parser.add_argument('--parallel', action='store_true', help='Enable parallel indexing.')
    index_parser.add_argument('--workers', type=int, default=4, help='Number of worker threads for parallel indexing.')

    # Query command
    query_parser = subparsers.add_parser('query', help='Query the indexed summaries.')
    query_parser.add_argument('root', nargs='?', default='.', type=Path, help='Root directory of the indexed summaries.')
    query_parser.add_argument('query', help='Query string.')
    query_parser.add_argument('--max_depth', type=int, default=10, help='Maximum depth for BFS traversal.')

    # Interactive query command
    interactive_parser = subparsers.add_parser('interactive', help='Interactive querying mode.')
    interactive_parser.add_argument('root', nargs='?', default='.', type=Path, help='Root directory of the indexed summaries.')
    interactive_parser.add_argument('--max_depth', type=int, default=10, help='Maximum depth for BFS traversal.')

    # Clean command
    clean_parser = subparsers.add_parser('clean', help='Remove .summarai.json files from the directory tree.')
    clean_parser.add_argument('root', nargs='?', default='.', type=Path, help='Root directory to clean.')

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.debug)

    # Initialize PromptManager
    prompt_manager = PromptManager()

    # Initialize LLM based on arguments
    if args.llm == 'openai':
        raise NotImplementedError("OpenAI API is not available in this version.")
    elif args.llm == 'mock':
        base_llm = Mock_LLM(prompt_manager=prompt_manager, debug=args.debug)
    elif args.llm == 'ollama':
        base_llm = Ollama_LLM(api_url="http://localhost:11434", prompt_manager=prompt_manager, debug=args.debug)
    else:
        logger.error(f"Unsupported LLM type: {args.llm}")
        return

    if args.cache:
        base_llm = Caching_LLM(base_llm, cache_file=Path('llm_cache.json'), debug=args.debug)

    # Wrap with Logging_LLM if debug is enabled
    if args.debug:
        base_llm = Logging_LLM(base_llm, debug=args.debug)

    # Handle commands
    if args.command == 'index':
        root = args.root.resolve()
        if args.parallel:
            logger.info(f"Starting parallel indexing of directory: {root}")
            index_directory_tree_parallel(root, base_llm, max_workers=args.workers)
        else:
            logger.info(f"Starting indexing of directory: {root}")
            index_directory_tree(root, base_llm)
    elif args.command == 'query':
        root = args.root.resolve()
        logger.info(f"Starting query in directory: {root} with query: '{args.query}'")
        results = query_summarai(root, args.query, base_llm, max_depth=args.max_depth)
        if results:
            print("\nRelevant files:")
            for file in results:
                print(file)
        else:
            print("\nNo relevant files found.")
    elif args.command == 'interactive':
        root = args.root.resolve()
        logger.info(f"Starting interactive query mode in directory: {root}")
        interactive_query(root, base_llm, max_depth=args.max_depth)
    elif args.command == 'clean':
        root = args.root.resolve()
        logger.info(f"Starting cleanup of directory: {root}")
        clean_directory_tree(root)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
