# SPDX-FileCopyrightText: 2026 WalshLab
# SPDX-License-Identifier: MIT

"""Tests for the BraiAn MCP Server AST parser."""

import sys
from pathlib import Path

import pytest

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from ast_parser import (
    build_package_index,
    categorize_function,
    extract_signature,
    get_package_path,
    parse_module,
)


class TestGetPackagePath:
    """Tests for get_package_path function."""

    def test_finds_installed_package(self):
        """Should find path for installed packages."""
        path = get_package_path("braian")
        assert path is not None
        assert path.exists()

    def test_returns_none_for_missing_package(self):
        """Should return None for non-existent packages."""
        path = get_package_path("nonexistent_package_xyz123")
        assert path is None


class TestParseModule:
    """Tests for parse_module function."""

    def test_parses_python_file(self):
        """Should parse a Python file and extract structure."""
        # Use the config.py file in this package as test input
        test_file = Path(__file__).parent.parent / "config.py"
        result = parse_module(test_file)

        assert "error" not in result
        assert "path" in result
        assert "classes" in result
        assert "functions" in result

    def test_handles_missing_file(self):
        """Should raise FileNotFoundError for missing files."""
        with pytest.raises(FileNotFoundError):
            parse_module(Path("/nonexistent/file.py"))


class TestCategorizeFunction:
    """Tests for categorize_function."""

    def test_categorizes_plotting_function(self):
        """Should detect plotting-related functions."""
        func_info = {
            "name": "plot_heatmap",
            "docstring": "Creates a visualization of the data.",
        }
        tags = categorize_function(func_info)
        assert "plotting" in tags

    def test_categorizes_statistics_function(self):
        """Should detect statistics-related functions."""
        func_info = {
            "name": "calculate_density",
            "docstring": "Computes the density metric.",
        }
        tags = categorize_function(func_info)
        assert "statistics" in tags

    def test_categorizes_brain_region_function(self):
        """Should detect brain region-related functions."""
        func_info = {
            "name": "get_region_info",
            "docstring": "Returns atlas annotation data.",
        }
        tags = categorize_function(func_info)
        assert "brain_region" in tags

    def test_returns_empty_for_generic_function(self):
        """Should return empty list for generic functions."""
        func_info = {
            "name": "helper",
            "docstring": "A utility function.",
        }
        tags = categorize_function(func_info)
        assert tags == []


class TestBuildPackageIndex:
    """Tests for build_package_index."""

    def test_indexes_braian_package(self):
        """Should successfully index the braian package."""
        index = build_package_index("braian")

        assert "error" not in index
        assert "modules" in index
        assert len(index["modules"]) > 0

    def test_returns_error_for_missing_package(self):
        """Should return error for non-existent packages."""
        index = build_package_index("nonexistent_package_xyz123")
        assert "error" in index


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
