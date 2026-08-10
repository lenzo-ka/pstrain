#!/usr/bin/env python3
"""Fix CMakeLists.txt files in programs to use pstrainc instead of sphinxtrain."""

from pathlib import Path

csrc = Path(__file__).parent.parent / "csrc" / "programs"

for cmake_file in csrc.rglob("CMakeLists.txt"):
    content = cmake_file.read_text()

    # Replace sphinxtrain with pstrainc
    content = content.replace("sphinxtrain", "pstrainc")

    # Fix include paths
    content = content.replace("${CMAKE_SOURCE_DIR}/include", "${CMAKE_SOURCE_DIR}/csrc/include")

    # Add csrc to private includes
    if "CMAKE_SOURCE_DIR}/csrc/include" in content:
        # Check if csrc is already there
        if "${CMAKE_SOURCE_DIR}/csrc" not in content:
            # Insert after the csrc/include line
            lines = content.split("\n")
            new_lines = []
            for i, line in enumerate(lines):
                new_lines.append(line)
                if "CMAKE_SOURCE_DIR}/csrc/include" in line and i + 1 < len(lines):
                    # Add csrc path on next line
                    indent = len(line) - len(line.lstrip())
                    new_lines.append(" " * indent + "${PROGRAM} PRIVATE ${CMAKE_SOURCE_DIR}/csrc")
            content = "\n".join(new_lines)

    # Fix install destination
    content = content.replace("${CMAKE_INSTALL_LIBEXECDIR}/sphinxtrain", "${CMAKE_INSTALL_BINDIR}")

    cmake_file.write_text(content)
    print(f"Fixed {cmake_file}")
