# SPDX-FileCopyrightText: 2026 WalshLab
# SPDX-License-Identifier: MIT

"""BraiAn AST MCP Server - Provides structured access to BraiAn and its dependencies."""

import json
import logging
import os
import sys
from typing import Any

from fastmcp import FastMCP

# Configure logging
logging.basicConfig(
    filename="braian_mcp.log",
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("braian_mcp")

# Ensure current working directory is in sys.path so we can find local packages
cwd = os.getcwd()
if cwd not in sys.path:
    sys.path.insert(0, cwd)
logger.info(f"Server started. CWD: {cwd}")
logger.info(f"sys.path: {sys.path}")

from .ast_parser import (
    build_package_index,
    categorize_function,
    get_package_path,
)
from .config import ALL_PACKAGES

# Create the MCP server
mcp = FastMCP(
    name="braian-ast",
    instructions="""
    This MCP server provides structured access to the BraiAn codebase and its dependencies.
    
    Use it to:
    - Explore package structure and modules
    - Find classes and their methods
    - Search for plotting and statistical functions
    - Get function signatures and docstrings
    """
)

# Cache for package indices
_package_cache: dict[str, dict] = {}


def _get_index(package_name: str) -> dict:
    """Get or build the index for a package."""
    logger.debug(f"Getting index for {package_name}")
    if package_name not in _package_cache:
        try:
            logger.info(f"Building index for {package_name}...")
            _package_cache[package_name] = build_package_index(package_name)
            logger.info(f"Built index for {package_name}. Modules: {len(_package_cache[package_name].get('modules', []))}")
        except Exception as e:
            logger.error(f"Failed to build index for {package_name}: {e}", exc_info=True)
            _package_cache[package_name] = {"error": str(e), "modules": []}
            
    return _package_cache[package_name]


def _get_all_indices() -> dict[str, dict]:
    """Get indices for all configured packages."""
    for pkg in ALL_PACKAGES:
        _get_index(pkg)
    return _package_cache


# ============================================================================
# LOGIC HELPERS
# ============================================================================

def _list_packages_logic() -> str:
    """Logic for listing packages."""
    packages = []
    for pkg_name in ALL_PACKAGES:
        pkg_path = get_package_path(pkg_name)
        packages.append({
            "name": pkg_name,
            "path": str(pkg_path) if pkg_path else None,
            "available": pkg_path is not None,
        })
    return json.dumps(packages, indent=2)


def _list_modules_logic(package_name: str) -> str:
    """Logic for listing modules."""
    index = _get_index(package_name)
    if "error" in index:
        return json.dumps({"error": index["error"]})
    
    modules = []
    for mod in index.get("modules", []):
        modules.append({
            "name": mod["name"],
            "classes": [c["name"] for c in mod.get("classes", [])],
            "functions": [f["name"] for f in mod.get("functions", [])],
        })
    return json.dumps(modules, indent=2)


def _get_class_logic(package_name: str, module_name: str, class_name: str) -> str:
    """Logic for getting class info."""
    index = _get_index(package_name)
    if "error" in index:
        return json.dumps({"error": index["error"]})
    
    for mod in index.get("modules", []):
        if mod["name"] == module_name or module_name == "*":
            for cls in mod.get("classes", []):
                if cls["name"] == class_name:
                    return json.dumps(cls, indent=2)
    
    return json.dumps({"error": f"Class '{class_name}' not found in {package_name}.{module_name}"})


def _get_function_logic(package_name: str, module_name: str, function_name: str) -> str:
    """Logic for getting function info."""
    index = _get_index(package_name)
    if "error" in index:
        return json.dumps({"error": index["error"]})
    
    for mod in index.get("modules", []):
        if mod["name"] == module_name or module_name == "*":
            for func in mod.get("functions", []):
                if func["name"] == function_name:
                    return json.dumps(func, indent=2)
            # Also check class methods
            for cls in mod.get("classes", []):
                for method in cls.get("methods", []) + cls.get("properties", []):
                    if method["name"] == function_name:
                        result = dict(method)
                        result["class"] = cls["name"]
                        return json.dumps(result, indent=2)
    
    return json.dumps({"error": f"Function '{function_name}' not found in {package_name}.{module_name}"})


def _list_plotting_functions_logic() -> str:
    """Logic for listing plotting functions."""
    results = []
    for pkg_name in ALL_PACKAGES:
        index = _get_index(pkg_name)
        for mod in index.get("modules", []):
            for func in mod.get("functions", []):
                tags = categorize_function(func)
                if "plotting" in tags:
                    results.append({
                        "package": pkg_name,
                        "module": mod["name"],
                        "name": func["name"],
                        "signature": func["signature"],
                        "docstring": func["docstring"][:200] if func["docstring"] else "",
                    })
    return json.dumps(results, indent=2)


def _list_statistics_functions_logic() -> str:
    """Logic for listing statistics functions."""
    results = []
    for pkg_name in ALL_PACKAGES:
        index = _get_index(pkg_name)
        for mod in index.get("modules", []):
            for func in mod.get("functions", []):
                tags = categorize_function(func)
                if "statistics" in tags:
                    results.append({
                        "package": pkg_name,
                        "module": mod["name"],
                        "name": func["name"],
                        "signature": func["signature"],
                        "docstring": func["docstring"][:200] if func["docstring"] else "",
                    })
    return json.dumps(results, indent=2)


# ============================================================================
# RESOURCES
# ============================================================================

@mcp.resource("braian://packages")
def list_packages() -> str:
    """List all indexed packages with their paths."""
    return _list_packages_logic()


@mcp.resource("braian://package/{package_name}/modules")
def list_modules(package_name: str) -> str:
    """List all modules in a package."""
    return _list_modules_logic(package_name)


@mcp.resource("braian://class/{package_name}/{module_name}/{class_name}")
def get_class(package_name: str, module_name: str, class_name: str) -> str:
    """Get detailed information about a class."""
    return _get_class_logic(package_name, module_name, class_name)


@mcp.resource("braian://function/{package_name}/{module_name}/{function_name}")
def get_function(package_name: str, module_name: str, function_name: str) -> str:
    """Get detailed information about a function."""
    return _get_function_logic(package_name, module_name, function_name)


@mcp.resource("braian://plotting")
def list_plotting_functions() -> str:
    """List all plotting-related functions across indexed packages."""
    return _list_plotting_functions_logic()


@mcp.resource("braian://statistics")
def list_statistics_functions() -> str:
    """List all statistics-related functions across indexed packages."""
    return _list_statistics_functions_logic()


# ============================================================================
# TOOLS
# ============================================================================

@mcp.tool()
def search_code(query: str, package: str | None = None) -> str:
    """Search for functions and classes by name pattern.
    
    Args:
        query: Search pattern (case-insensitive substring match).
        package: Optional package name to limit search.
        
    Returns:
        JSON list of matching items.
    """
    logger.info(f"Tool call: search_code(query='{query}', package='{package}')")
    query_lower = query.lower()
    results = []
    
    packages_to_search = [package] if package else ALL_PACKAGES
    
    for pkg_name in packages_to_search:
        index = _get_index(pkg_name)
        if "error" in index:
            logger.warning(f"Skipping search in {pkg_name} due to index error: {index['error']}")
            continue
            
        for mod in index.get("modules", []):
            # Search functions
            for func in mod.get("functions", []):
                if query_lower in func["name"].lower():
                    results.append({
                        "type": "function",
                        "package": pkg_name,
                        "module": mod["name"],
                        "name": func["name"],
                        "signature": func["signature"],
                    })
            
            # Search classes
            for cls in mod.get("classes", []):
                if query_lower in cls["name"].lower():
                    results.append({
                        "type": "class",
                        "package": pkg_name,
                        "module": mod["name"],
                        "name": cls["name"],
                        "methods_count": len(cls.get("methods", [])),
                    })
                
                # Search methods
                for method in cls.get("methods", []):
                    if query_lower in method["name"].lower():
                        results.append({
                            "type": "method",
                            "package": pkg_name,
                            "module": mod["name"],
                            "class": cls["name"],
                            "name": method["name"],
                            "signature": method["signature"],
                        })
    
    logger.info(f"Found {len(results)} results")
    return json.dumps(results, indent=2)


@mcp.tool()
def get_class_methods(package_name: str, class_name: str) -> str:
    """Get all methods of a specific class.
    
    Args:
        package_name: Package containing the class.
        class_name: Name of the class.
        
    Returns:
        JSON with class info and all methods.
    """
    index = _get_index(package_name)
    if "error" in index:
        return json.dumps({"error": index["error"]})
    
    for mod in index.get("modules", []):
        for cls in mod.get("classes", []):
            if cls["name"] == class_name:
                return json.dumps({
                    "name": cls["name"],
                    "module": mod["name"],
                    "docstring": cls["docstring"],
                    "bases": cls.get("bases", []),
                    "methods": [
                        {"name": m["name"], "signature": m["signature"], "docstring": m["docstring"][:100]}
                        for m in cls.get("methods", [])
                    ],
                    "properties": [p["name"] for p in cls.get("properties", [])],
                }, indent=2)
    
    return json.dumps({"error": f"Class '{class_name}' not found in {package_name}"})


@mcp.tool()
def get_function_signature(package_name: str, function_name: str) -> str:
    """Get the full signature and docstring for a function.
    
    Args:
        package_name: Package containing the function.
        function_name: Name of the function.
        
    Returns:
        JSON with function details.
    """
    index = _get_index(package_name)
    if "error" in index:
        return json.dumps({"error": index["error"]})
    
    for mod in index.get("modules", []):
        for func in mod.get("functions", []):
            if func["name"] == function_name:
                return json.dumps({
                    "name": func["name"],
                    "module": mod["name"],
                    "signature": func["signature"],
                    "docstring": func["docstring"],
                    "decorators": func.get("decorators", []),
                }, indent=2)
    
    return json.dumps({"error": f"Function '{function_name}' not found in {package_name}"})


@mcp.tool()
def find_plotting_functions() -> str:
    """Find all visualization-related functions.
    
    Returns:
        JSON list of plotting functions with their signatures.
    """
    return _list_plotting_functions_logic()


@mcp.tool()
def find_statistical_functions() -> str:
    """Find all statistics and metrics functions.
    
    Returns:
        JSON list of statistical functions with their signatures.
    """
    return _list_statistics_functions_logic()


@mcp.tool()
def find_brain_region_functions() -> str:
    """Find functions related to brain regions and atlases.
    
    Returns:
        JSON list of brain region-related functions.
    """
    results = []
    for pkg_name in ALL_PACKAGES:
        index = _get_index(pkg_name)
        for mod in index.get("modules", []):
            for func in mod.get("functions", []):
                tags = categorize_function(func)
                if "brain_region" in tags:
                    results.append({
                        "package": pkg_name,
                        "module": mod["name"],
                        "name": func["name"],
                        "signature": func["signature"],
                        "docstring": func["docstring"][:200] if func["docstring"] else "",
                    })
    return json.dumps(results, indent=2)


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    """Run the MCP server."""
    if "--test" in sys.argv:
        # Quick test mode
        print("Testing BraiAn AST MCP Server...")
        print(f"Configured packages: {ALL_PACKAGES}")
        print("\nIndexing packages...")
        for pkg in ALL_PACKAGES:
            index = build_package_index(pkg)
            if "error" in index:
                print(f"  ❌ {pkg}: {index['error']}")
            else:
                mod_count = len(index.get("modules", []))
                print(f"  ✅ {pkg}: {mod_count} modules indexed")
        print("\nTest complete!")
        return
    
    # Run the server
    mcp.run()


if __name__ == "__main__":
    main()
