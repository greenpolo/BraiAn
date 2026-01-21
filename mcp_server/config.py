# SPDX-FileCopyrightText: 2026 WalshLab
# SPDX-License-Identifier: MIT

"""Configuration for the BraiAn AST MCP Server."""

# Core packages (always indexed)
CORE_PACKAGES = [
    "braian",
]

# Novel/specialized dependencies (always indexed)
NOVEL_PACKAGES = [
    "igraph",
    "treelib",
    "brainglobe_atlasapi",
    "brainglobe_heatmap",
    "brainglobe_space",
    "brainglobe_utils",
]

# Combine all packages to index
ALL_PACKAGES = CORE_PACKAGES + NOVEL_PACKAGES
