# SPDX-FileCopyrightText: 2026 WalshLab
# SPDX-License-Identifier: MIT

"""AST parsing utilities for extracting code structure from Python packages."""

import ast
import importlib.util
import inspect
import sys
from pathlib import Path
from typing import Any


def get_package_path(package_name: str) -> Path | None:
    """Get the file path of an installed package.
    
    Args:
        package_name: Name of the package to locate.
        
    Returns:
        Path to the package directory, or None if not found.
    """
    try:
        spec = importlib.util.find_spec(package_name)
        if spec and spec.origin:
            origin = Path(spec.origin)
            # If it's a package, return parent; if module, return the file
            if origin.name == "__init__.py":
                return origin.parent
            return origin
        elif spec and spec.submodule_search_locations:
            # Package without __init__.py
            return Path(spec.submodule_search_locations[0])
    except (ModuleNotFoundError, ValueError):
        pass
    return None


def extract_docstring(node: ast.AST) -> str:
    """Extract docstring from an AST node.
    
    Args:
        node: AST node (FunctionDef, ClassDef, or Module).
        
    Returns:
        Docstring if present, empty string otherwise.
    """
    return ast.get_docstring(node) or ""


def extract_signature(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    """Extract function signature from AST node.
    
    Args:
        node: FunctionDef or AsyncFunctionDef node.
        
    Returns:
        String representation of the function signature.
    """
    args = node.args
    parts = []
    
    # Positional-only args
    for arg in args.posonlyargs:
        parts.append(_format_arg(arg))
    if args.posonlyargs:
        parts.append("/")
    
    # Regular positional args
    defaults_offset = len(args.args) - len(args.defaults)
    for i, arg in enumerate(args.args):
        default_idx = i - defaults_offset
        default = args.defaults[default_idx] if default_idx >= 0 else None
        parts.append(_format_arg(arg, default))
    
    # *args
    if args.vararg:
        parts.append(f"*{args.vararg.arg}")
    elif args.kwonlyargs:
        parts.append("*")
    
    # Keyword-only args
    for i, arg in enumerate(args.kwonlyargs):
        default = args.kw_defaults[i]
        parts.append(_format_arg(arg, default))
    
    # **kwargs
    if args.kwarg:
        parts.append(f"**{args.kwarg.arg}")
    
    sig = f"({', '.join(parts)})"
    
    # Return annotation
    if node.returns:
        sig += f" -> {_unparse_annotation(node.returns)}"
    
    return sig


def _format_arg(arg: ast.arg, default: ast.expr | None = None) -> str:
    """Format a function argument."""
    result = arg.arg
    if arg.annotation:
        result += f": {_unparse_annotation(arg.annotation)}"
    if default is not None:
        try:
            result += f"={ast.unparse(default)}"
        except Exception:
            result += "=..."
    return result


def _unparse_annotation(node: ast.expr) -> str:
    """Safely unparse an annotation."""
    try:
        return ast.unparse(node)
    except Exception:
        return "..."


def parse_module(path: Path) -> dict[str, Any]:
    """Parse a Python module and extract its structure.
    
    Args:
        path: Path to the Python file.
        
    Returns:
        Dictionary containing module info: classes, functions, imports.
    """
    try:
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
    except (SyntaxError, UnicodeDecodeError) as e:
        return {"error": str(e), "classes": [], "functions": [], "imports": []}
    
    result = {
        "path": str(path),
        "docstring": extract_docstring(tree),
        "classes": [],
        "functions": [],
        "imports": [],
    }
    
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.ClassDef):
            result["classes"].append(_parse_class(node))
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            result["functions"].append(_parse_function(node))
        elif isinstance(node, ast.Import):
            for alias in node.names:
                result["imports"].append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                result["imports"].append(node.module)
    
    return result


def _parse_class(node: ast.ClassDef) -> dict[str, Any]:
    """Parse a class definition."""
    methods = []
    properties = []
    
    for item in node.body:
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
            func_info = _parse_function(item)
            # Check for @property decorator
            is_property = any(
                (isinstance(d, ast.Name) and d.id == "property") or
                (isinstance(d, ast.Attribute) and d.attr == "property")
                for d in item.decorator_list
            )
            if is_property:
                properties.append(func_info)
            else:
                methods.append(func_info)
    
    bases = [ast.unparse(b) for b in node.bases]
    
    return {
        "name": node.name,
        "docstring": extract_docstring(node),
        "bases": bases,
        "methods": methods,
        "properties": properties,
        "line": node.lineno,
    }


def _parse_function(node: ast.FunctionDef | ast.AsyncFunctionDef) -> dict[str, Any]:
    """Parse a function definition."""
    is_async = isinstance(node, ast.AsyncFunctionDef)
    decorators = []
    for d in node.decorator_list:
        try:
            decorators.append(ast.unparse(d))
        except Exception:
            decorators.append("...")
    
    return {
        "name": node.name,
        "signature": extract_signature(node),
        "docstring": extract_docstring(node),
        "decorators": decorators,
        "is_async": is_async,
        "line": node.lineno,
    }


def build_package_index(package_name: str) -> dict[str, Any]:
    """Build an index of a package's structure.
    
    Args:
        package_name: Name of the package to index.
        
    Returns:
        Dictionary containing the package structure.
    """
    pkg_path = get_package_path(package_name)
    if pkg_path is None:
        return {"error": f"Package '{package_name}' not found", "modules": []}
    
    result = {
        "name": package_name,
        "path": str(pkg_path),
        "modules": [],
    }
    
    if pkg_path.is_file():
        # Single module package
        module_info = parse_module(pkg_path)
        module_info["name"] = package_name
        result["modules"].append(module_info)
    else:
        # Package with multiple modules
        for py_file in pkg_path.rglob("*.py"):
            if py_file.name.startswith("__pycache__"):
                continue
            
            # Calculate module name relative to package
            rel_path = py_file.relative_to(pkg_path)
            parts = list(rel_path.parts)
            if parts[-1] == "__init__.py":
                parts = parts[:-1]
            else:
                parts[-1] = parts[-1][:-3]  # Remove .py
            
            module_name = ".".join(parts) if parts else package_name
            
            module_info = parse_module(py_file)
            module_info["name"] = module_name
            result["modules"].append(module_info)
    
    return result


def categorize_function(func_info: dict) -> list[str]:
    """Categorize a function based on its name and docstring.
    
    Args:
        func_info: Function info dictionary from _parse_function.
        
    Returns:
        List of category tags (e.g., ["plotting", "statistics"]).
    """
    tags = []
    name = func_info.get("name", "").lower()
    docstring = func_info.get("docstring", "").lower()
    combined = name + " " + docstring
    
    # Plotting keywords
    plot_keywords = ["plot", "graph", "chart", "viz", "heatmap", "figure", "draw", "xmas_tree"]
    if any(kw in combined for kw in plot_keywords):
        tags.append("plotting")
    
    # Statistics keywords
    stats_keywords = ["density", "percentage", "fold_change", "correlation", "metric", 
                      "jaccard", "similarity", "overlap", "pls", "statistic", "test"]
    if any(kw in combined for kw in stats_keywords):
        tags.append("statistics")
    
    # Brain region keywords
    brain_keywords = ["region", "ontology", "atlas", "hemisphere", "brain", "annotation"]
    if any(kw in combined for kw in brain_keywords):
        tags.append("brain_region")
    
    return tags
