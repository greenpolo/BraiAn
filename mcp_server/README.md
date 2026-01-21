# BraiAn MCP Server

An MCP (Model Context Protocol) server that provides structured access to the BraiAn codebase and its dependencies through AST parsing.

## Features

- **Code Search**: Find functions, classes, and methods by name pattern
- **Function Signatures**: Get full signatures and docstrings for any function
- **Class Exploration**: List all methods and properties of a class
- **Specialized Queries**: Find plotting, statistical, and brain-region functions

## Installation

```bash
# From the mcp_server directory
pip install -e .
```

## Usage

### Running the Server

```bash
# Via the installed script
braian-mcp

# Or directly with Python
python -m mcp_server.braian_mcp_server
```

### Testing

```bash
# Quick test to verify indexing works
python -m mcp_server.braian_mcp_server --test
```

## Available Tools

| Tool | Description |
|------|-------------|
| `search_code` | Search for functions/classes by name pattern |
| `get_class_methods` | Get all methods of a specific class |
| `get_function_signature` | Get signature and docstring for a function |
| `find_plotting_functions` | Find all visualization-related functions |
| `find_statistical_functions` | Find all statistics/metrics functions |
| `find_brain_region_functions` | Find brain region and atlas functions |

## Configuration

Edit `config.py` to modify which packages are indexed:

- `CORE_PACKAGES`: Always indexed (braian)
- `NOVEL_PACKAGES`: Specialized dependencies (igraph, treelib, brainglobe_*)

## License

MIT License - See LICENSE file for details.
