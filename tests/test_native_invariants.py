"""Structural regression gates for native-process invariants."""

from __future__ import annotations

import ast
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
LIBPSTRAIN = ROOT / "csrc" / "libs" / "libpstrain"
NATIVE_EXTENSIONS = {".c", ".cc", ".cpp", ".cxx"}

# Native stdout exceptions are scoped to (file, function, exact call). Add one
# only after proving that every route to that function redirects fd 1, and add
# both a focused scanner test and a behavioral mutation that fails without it.
#
# Known limits, intentionally out of scope for this accidental-regression gate:
# - C preprocessor token-pasting can synthesize a guarded name; chasing expanded
#   tokens would require a compiler-aware gate disproportionate to this check.
# - Runtime symbol lookup or cast-obscured C calls are not resolved; those are
#   deliberate indirection rather than a plausible accidental progress print.
# - A stream made unsafe only by an assignment after the call in a loop is not
#   modeled; recognizing loop-carried assignments would require C dataflow.
# - Python runtime monkeypatching or metaclass-generated pools are not modeled;
#   production code must keep pool construction statically visible instead.
# - Function-local imports can shadow a module alias in this early static check;
#   the authoritative runtime constructor guard is scope-aware by construction.
#
# These progress rows are safe because every production route redirects them;
# the behavioral mini-Arctic gate independently covers that redirection.
ALLOWED_NATIVE_STDOUT_CALLS = Counter(
    {
        (
            "csrc/libs/libpstrain/pstrain_bw.c",
            "pstrain_bw_process_utt_mfcc",
            'printf("utt> %5u %25s %4u", ctx->total_utts + 1, utterance_id, n_mfcc_frames);',
        ),
        (
            "csrc/libs/libpstrain/pstrain_bw.c",
            "pstrain_bw_process_utt_mfcc",
            'printf(" %e\\n", log_forw_prob);',
        ),
        (
            "csrc/libs/libpstrain/pstrain_bw.c",
            "pstrain_bw_process_utt_mfcc",
            'printf(" failed\\n");',
        ),
    }
)

# This pool makes each worker the native containment boundary, so its work runs
# native code directly instead of starting a native helper child.
POOLS_WITHOUT_NATIVE_HELPERS = {
    # This exception matches the initializer name, not every caller of the
    # public BWTrainer.process_utterance_mfcc method. A future caller outside
    # train.py must be audited separately because it could expose fd 1.
    ("pstrain/lib/steps/train.py", "run_bw_training"): ("_initialize_bw_pool_worker",),
}

POOLS_THAT_INSTALL_HELPER_FINALIZER = {
    ("pstrain/lib/pipeline/runner.py", "_run_parallel_batch"): ("_initialize_pool_worker",),
    (
        "pstrain/lib/testing/test.py",
        "_decode_files",
    ): ("pstrain.lib.native_worker.close_helper_before_children_are_joined",),
}

ALWAYS_STDOUT_FUNCTIONS = {
    "printf",
    "puts",
    "putchar",
    "putchar_unlocked",
    "vprintf",
}
STREAM_FUNCTION_ARGUMENTS = {
    "fprintf": 0,
    "vfprintf": 0,
    "fputs": 1,
    "fputc": 1,
    "putc": 1,
    "fwrite": 3,
    "fwprintf": 0,
    "vfwprintf": 0,
    "fputwc": 1,
    "putwc": 1,
    "putc_unlocked": 1,
    "fputs_unlocked": 1,
    "fwrite_unlocked": 3,
}
FD_FUNCTIONS = {"write", "_write", "writev", "pwrite", "dprintf", "vdprintf"}


def _without_c_comments(source: str) -> str:
    """Remove comments without treating comment markers inside strings as syntax."""
    output: list[str] = []
    index = 0
    quote: str | None = None
    while index < len(source):
        char = source[index]
        following = source[index + 1] if index + 1 < len(source) else ""
        if quote is not None:
            output.append(char)
            if char == "\\" and following:
                output.append(following)
                index += 2
                continue
            if char == quote:
                quote = None
            index += 1
            continue
        if char in {'"', "'"}:
            quote = char
            output.append(char)
            index += 1
            continue
        if char == "/" and following == "/":
            index = source.find("\n", index)
            if index == -1:
                break
            output.append("\n")
            index += 1
            continue
        if char == "/" and following == "*":
            end = source.find("*/", index + 2)
            if end == -1:
                raise AssertionError("unterminated C comment in scanned source")
            output.extend("\n" for char in source[index : end + 2] if char == "\n")
            index = end + 2
            continue
        output.append(char)
        index += 1
    return "".join(output)


def _without_c_strings(source: str) -> str:
    """Blank string and character literals while preserving source positions."""
    output = list(source)
    quote: str | None = None
    index = 0
    while index < len(source):
        char = source[index]
        if quote is not None:
            if char != "\n":
                output[index] = " "
            if char == "\\" and index + 1 < len(source):
                index += 1
                if source[index] != "\n":
                    output[index] = " "
            elif char == quote:
                quote = None
        elif char in {'"', "'"}:
            quote = char
            output[index] = " "
        index += 1
    if quote is not None:
        raise AssertionError("unterminated C string in scanned source")
    return "".join(output)


def _split_c_arguments(arguments: str) -> list[str]:
    parts: list[str] = []
    start = 0
    depth = 0
    quote: str | None = None
    index = 0
    while index < len(arguments):
        char = arguments[index]
        if quote is not None:
            if char == "\\":
                index += 2
                continue
            if char == quote:
                quote = None
        elif char in {'"', "'"}:
            quote = char
        elif char in "([{":
            depth += 1
        elif char in ")]}":
            depth -= 1
        elif char == "," and depth == 0:
            parts.append(arguments[start:index].strip())
            start = index + 1
        index += 1
    parts.append(arguments[start:].strip())
    return parts


def _c_calls(source: str) -> list[tuple[str, list[str], str, int]]:
    source = _without_c_comments(source)
    syntax = _without_c_strings(source)
    calls: list[tuple[str, list[str], str, int]] = []
    names = ALWAYS_STDOUT_FUNCTIONS | set(STREAM_FUNCTION_ARGUMENTS) | FD_FUNCTIONS
    # Only bare C callees count. Member calls such as obj.write() and
    # ptr->write() are not the C library functions guarded here.
    pattern = re.compile(rf"(?<![.>\w])({'|'.join(sorted(names))})\s*\(")
    for match in pattern.finditer(syntax):
        depth = 1
        index = match.end()
        while index < len(syntax) and depth:
            char = syntax[index]
            if char == "(":
                depth += 1
            elif char == ")":
                depth -= 1
            index += 1
        if depth:
            raise AssertionError(f"unterminated {match.group(1)} call in scanned source")
        end = index
        while end < len(source) and source[end].isspace():
            end += 1
        if end < len(source) and source[end] == "{":
            # A definition of a project-local colliding name is not itself a
            # call. Any invocation is still treated as the guarded C function.
            continue
        # A guarded call may be part of a condition, return expression, or
        # macro replacement. Inventory the call itself; an exact exception can
        # then be scoped to its file and function like any statement call.
        if end < len(source) and source[end] == ";":
            end += 1
        text = " ".join(source[match.start() : end].split())
        calls.append(
            (
                match.group(1),
                _split_c_arguments(source[match.end() : index - 1]),
                text,
                match.start(),
            )
        )
    return calls


def _c_function_bounds(source: str, position: int) -> tuple[int, int]:
    """Return bounds of the outermost brace scope containing a call."""
    syntax = _without_c_strings(source)
    stack: list[int] = []
    for index, char in enumerate(syntax[:position]):
        if char == "{":
            stack.append(index)
        elif char == "}" and stack:
            stack.pop()
    if not stack:
        return (0, len(source))
    start = stack[0]
    depth = 1
    for end in range(start + 1, len(syntax)):
        if syntax[end] == "{":
            depth += 1
        elif syntax[end] == "}":
            depth -= 1
            if depth == 0:
                return (start, end + 1)
    raise AssertionError("unterminated C function scope in scanned source")


def _c_function_name(source: str, position: int) -> str:
    start, _ = _c_function_bounds(source, position)
    if start == 0:
        return "<global>"
    prefix = source[:start]
    match = re.search(r"\b([A-Za-z_]\w*)\s*\([^;{}]*\)\s*$", prefix)
    if match is None:
        raise AssertionError("could not identify enclosing C function")
    return match.group(1)


def _stream_is_provably_safe(stream: str, source: str, position: int) -> bool:
    stream = stream.strip().strip("()")
    if stream == "stderr":
        return True
    if not re.fullmatch(r"[A-Za-z_]\w*", stream):
        return False
    start, _ = _c_function_bounds(source, position)
    reaching_prefix = source[start:position]
    assignments = list(
        re.finditer(
            rf"(?:\bFILE\s*\*\s*)?\b{re.escape(stream)}\s*=\s*([^;]+);",
            reaching_prefix,
        )
    )
    if not assignments:
        return False

    def safe_opener(value: str) -> bool:
        opener = re.fullmatch(r"(fopen|fdopen|tmpfile|open_memstream)\s*\((.*)\)", value)
        if opener is None:
            return False
        arguments = _split_c_arguments(opener.group(2))
        if opener.group(1) == "fopen":
            return bool(arguments) and arguments[0].strip() != '"/dev/stdout"'
        if opener.group(1) != "fdopen":
            return True
        if not arguments:
            return False
        descriptor = re.sub(r"\([^)]*\)", "", arguments[0]).strip()
        return descriptor not in {"1", "STDOUT_FILENO"} and bool(
            re.fullmatch(r"(?:[02-9]|[1-9][0-9]+|STDERR_FILENO)", descriptor)
        )

    # Control flow is intentionally not modeled. Every assignment preceding
    # the call in its function must be safe, so a stdout default cannot be
    # hidden by a later conditional fopen override (or vice versa).
    return all(safe_opener(match.group(1).strip()) for match in assignments)


def _indirect_guarded_references(
    source: str, direct_call_positions: set[int]
) -> list[tuple[str, int]]:
    """Find guarded function names used through aliases or indirect syntax."""
    syntax = _without_c_strings(_without_c_comments(source))
    names = ALWAYS_STDOUT_FUNCTIONS | set(STREAM_FUNCTION_ARGUMENTS) | FD_FUNCTIONS
    references: list[tuple[str, int]] = []
    for match in re.finditer(rf"\b({'|'.join(sorted(names))})\b", syntax):
        if match.start() in direct_call_positions:
            continue
        tail = syntax[match.end() :]
        if re.match(r"\s*\(", tail):
            close = tail.find(")")
            following = tail[close + 1 :].lstrip() if close != -1 else ""
            if following.startswith(("{", ";")):
                continue
        references.append((match.group(1), match.start()))
    return references


def _native_stdout_calls_in(source: str) -> Counter[str]:
    source = _without_c_comments(source)
    calls: Counter[str] = Counter()
    direct_calls = _c_calls(source)
    for function, arguments, text, position in direct_calls:
        if function in ALWAYS_STDOUT_FUNCTIONS:
            calls[text] += 1
        elif function in FD_FUNCTIONS and arguments:
            descriptor = re.sub(r"\([^)]*\)", "", arguments[0]).strip()
            if descriptor in {"1", "STDOUT_FILENO"}:
                calls[text] += 1
        else:
            stream_argument = STREAM_FUNCTION_ARGUMENTS.get(function)
            if (
                stream_argument is not None
                and len(arguments) > stream_argument
                # Deliberately over-strict: an unknown FILE* may be stdout, and
                # that failure is expensive and silent. Only stderr or a genuine
                # file handle opened in this same function is provably safe.
                and not _stream_is_provably_safe(arguments[stream_argument], source, position)
            ):
                calls[text] += 1
    for function, _ in _indirect_guarded_references(
        source, {position for _, _, _, position in direct_calls}
    ):
        calls[f"indirect reference to {function}"] += 1
    return calls


def _assert_native_stdout_allowed(source: str, allowed: Counter[str]) -> None:
    assert _native_stdout_calls_in(source) == allowed


def _cmake_libpstrain_sources() -> set[Path]:
    # This extracts literal libpstrain source entries from csrc/CMakeLists.txt;
    # it does not evaluate variables, generated sources, conditionals, or
    # included CMake files. Any disagreement with recursive discovery fails
    # closed so such a refactor must make its coverage explicit here.
    cmake = (ROOT / "csrc" / "CMakeLists.txt").read_text(encoding="utf-8")
    entries = re.findall(
        r"^\s*(libs/libpstrain/\S+\.(?:c|cc|cpp|cxx))\s*$", cmake, flags=re.MULTILINE
    )
    return {ROOT / "csrc" / entry for entry in entries}


def _discovered_libpstrain_sources() -> set[Path]:
    return {path for path in LIBPSTRAIN.rglob("*") if path.suffix.lower() in NATIVE_EXTENSIONS}


def _native_stdout_calls() -> Counter[tuple[str, str, str]]:
    calls: Counter[tuple[str, str, str]] = Counter()
    for path in sorted(_discovered_libpstrain_sources()):
        source = _without_c_comments(path.read_text(encoding="utf-8"))
        relative_path = path.relative_to(ROOT).as_posix()
        direct_calls = _c_calls(source)
        for _, _, text, position in direct_calls:
            if _native_stdout_calls_in(source)[text]:
                calls[(relative_path, _c_function_name(source, position), text)] += 1
        indirect = _indirect_guarded_references(
            source, {position for _, _, _, position in direct_calls}
        )
        for function, position in indirect:
            calls[
                (
                    relative_path,
                    _c_function_name(source, position),
                    f"indirect reference to {function}",
                )
            ] += 1
    return calls


@pytest.mark.parametrize(
    "statement",
    [
        'printf("progress");',
        'puts("progress");',
        "putchar(10);",
        "putchar_unlocked(10);",
        "putc_unlocked(10, stdout);",
        'vprintf("%s", args);',
        'write(1, "progress", 8);',
        '_write(1, "progress", 8);',
        'write(STDOUT_FILENO, "progress", 8);',
        'dprintf(1, "progress");',
        'vdprintf(1, "%s", args);',
        'fprintf(stdout, "progress");',
        'vfprintf(stdout, "%s", args);',
        'fputs("progress", stdout);',
        "fputc(10, stdout);",
        "putc(10, stdout);",
        "fwrite(data, 1, size, stdout);",
        'fputs_unlocked("progress", stdout);',
        "fwrite_unlocked(data, 1, size, stdout);",
        "writev(1, vectors, count);",
        'pwrite(1, "progress", 8, 0);',
        'fwprintf(stdout, L"progress");',
        'vfwprintf(stdout, L"%s", args);',
        "fputwc(L'x', stdout);",
        "putwc(L'x', stdout);",
        'FILE *out = stdout; fprintf(out, "progress");',
        'FILE *out; out = stdout; fputs("progress", out);',
        "FILE *out = stdout; FILE *copy = out; fwrite(data, 1, size, copy);",
    ],
)
def test_native_stdout_scanner_rejects_each_supported_spelling(statement: str) -> None:
    with pytest.raises(AssertionError):
        _assert_native_stdout_allowed(statement, Counter())


@pytest.mark.parametrize(
    "source",
    [
        'void emit(FILE *out) { fprintf(out, "x"); } void run(void) { emit(stdout); }',
        'void run(void) { fprintf(s.out, "x"); }',
        'void run(void) { fputs("x", streams[0]); }',
        'void run(void) { fputs("x", *out_ptr); }',
        'void run(void) { FILE *out = condition ? stdout : stderr; fprintf(out, "x"); }',
        'void run(void) { FILE *out = get_stdout(); fprintf(out, "x"); }',
    ],
    ids=["parameter", "struct-member", "array-member", "dereference", "ternary", "return"],
)
def test_unknown_stream_expressions_fail_closed(source: str) -> None:
    with pytest.raises(AssertionError):
        _assert_native_stdout_allowed(source, Counter())


@pytest.mark.parametrize(
    ("source", "violates"),
    [
        ('const char *s = "write(1, x, 1);";', False),
        ("void run(void) { obj.write(1, data, size); }", False),
        ("void write(int fd, void *data, int size) {} void run(void) { write(1, x, 1); }", True),
    ],
    ids=["string-literal", "member-call", "local-name-collision"],
)
def test_c_call_lexing_distinguishes_non_calls(source: str, violates: bool) -> None:
    if violates:
        with pytest.raises(AssertionError):
            _assert_native_stdout_allowed(source, Counter())
    else:
        _assert_native_stdout_allowed(source, Counter())


def test_stream_opened_in_same_function_is_provably_safe() -> None:
    source = 'void run(void) { FILE *out = fopen("result", "w"); fprintf(out, "x"); }'
    _assert_native_stdout_allowed(source, Counter())


def test_every_reaching_stream_assignment_must_be_safe() -> None:
    source = (
        "void run(const char *path) { FILE *out = stdout; "
        'if (path != NULL) { out = fopen(path, "w"); } fprintf(out, "x"); }'
    )
    with pytest.raises(AssertionError):
        _assert_native_stdout_allowed(source, Counter())


def test_fopen_dev_stdout_is_not_a_safe_handle() -> None:
    source = 'void run(void) { FILE *out = fopen("/dev/stdout", "w"); fprintf(out, "x"); }'
    with pytest.raises(AssertionError):
        _assert_native_stdout_allowed(source, Counter())


def test_guarded_call_in_expression_position_is_accepted() -> None:
    source = (
        'void run(void) { FILE *fp = fopen("result", "w"); '
        "if (fwrite(d, 1, n, fp) != n) { abort(); } }"
    )
    _assert_native_stdout_allowed(source, Counter())


def test_guarded_call_in_macro_can_use_scoped_exception() -> None:
    source = "#define LOGF(fp, ...) fprintf(fp, __VA_ARGS__)\n"
    calls = _native_stdout_calls_in(source)
    assert calls == Counter({"fprintf(fp, __VA_ARGS__)": 1})
    _assert_native_stdout_allowed(source, calls)


@pytest.mark.parametrize(
    "source",
    [
        'void run(void) { FILE *out = fdopen(1, "w"); fprintf(out, "x"); }',
        ('void run(void) { FILE *out = fopen("result", "w"); out = stdout; fprintf(out, "x"); }'),
    ],
    ids=["fdopen-stdout", "reassigned-to-stdout"],
)
def test_opened_stream_must_have_a_safe_reaching_assignment(source: str) -> None:
    with pytest.raises(AssertionError):
        _assert_native_stdout_allowed(source, Counter())


@pytest.mark.parametrize(
    "source",
    [
        'void run(void) { (write)(1, "x", 1); }',
        ('void run(void) { ssize_t (*emit)(int, const void *, size_t) = write; emit(1, "x", 1); }'),
        '#define EMIT write\nvoid run(void) { EMIT(1, "x", 1); }',
    ],
    ids=["parenthesized", "function-pointer", "macro-alias"],
)
def test_indirect_guarded_function_references_fail_closed(source: str) -> None:
    with pytest.raises(AssertionError):
        _assert_native_stdout_allowed(source, Counter())


def test_open_stream_name_from_another_function_does_not_taint_scope() -> None:
    source = (
        'void first(void) { FILE *out = fopen("result", "w"); } '
        'void second(FILE *out) { fprintf(out, "x"); }'
    )
    with pytest.raises(AssertionError):
        _assert_native_stdout_allowed(source, Counter())


def test_libpstrain_scan_matches_every_compiled_translation_unit() -> None:
    # Scan and build inventories must agree so a new native source cannot land
    # outside the fd-1 gate merely by using a new extension or subdirectory.
    assert _discovered_libpstrain_sources() == _cmake_libpstrain_sources()


def test_libpstrain_remains_c_only() -> None:
    # A textual C scanner cannot prove the absence of C++ iostream output.
    # Keep this library C-only so a new C++ translation unit fails closed.
    cpp_sources = sorted(
        path.relative_to(ROOT).as_posix()
        for path in _discovered_libpstrain_sources()
        if path.suffix.lower() != ".c"
    )
    assert not cpp_sources, (
        f"libpstrain must remain C-only; move C++ sources elsewhere in the tree: {cpp_sources}"
    )


def test_libpstrain_has_no_unredirected_stdout_calls() -> None:
    # Native training must never write to fd 1: a Jupyter capture pipe can fill
    # and park every worker in write(2), making training appear to hang.
    assert _native_stdout_calls() == ALLOWED_NATIVE_STDOUT_CALLS


def _name(node: ast.expr) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _name(node.value)
        return None if base is None else f"{base}.{node.attr}"
    return None


@dataclass(frozen=True)
class _PoolCall:
    owner: str
    initializer: str | None


class _PoolAnalyzer:
    """Provide an early, deliberately non-exhaustive process-pool inventory."""

    def __init__(
        self,
        source: str,
        package_pool_symbols: set[str] | None = None,
        module_name: str | None = None,
        package_nonfactory_functions: set[str] | None = None,
    ) -> None:
        self.tree = ast.parse(source)
        self.module_name = module_name
        self.package_nonfactory_functions = package_nonfactory_functions or set()
        self.aliases: dict[str, str] = {}
        self.pool_symbols: set[str] = {
            "concurrent.futures.ProcessPoolExecutor",
            "multiprocessing.Pool",
        }
        self.pool_symbols.update(package_pool_symbols or ())
        self.partial_symbols = {"functools.partial"}
        self._read_imports()
        self._resolve_pool_symbols()

    def _resolve_name(self, node: ast.expr) -> str | None:
        name = _name(node)
        if name is None:
            return None
        root, *suffix = name.split(".")
        resolved = self.aliases.get(root, root)
        return ".".join([resolved, *suffix])

    def _is_pool_symbol(self, node: ast.expr) -> bool:
        name = self._resolve_name(node)
        return name in self.pool_symbols if name is not None else False

    def _read_imports(self) -> None:
        for node in ast.walk(self.tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name in {"concurrent.futures", "multiprocessing"} and alias.asname:
                        raise AssertionError(
                            "pool-provider module aliased; import the pool constructor directly"
                        )
                    local = alias.asname or alias.name.split(".")[0]
                    self.aliases[local] = alias.name if alias.asname else local
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if node.level:
                    if self.module_name is None:
                        raise AssertionError("relative import requires a package module name")
                    package = self.module_name.split(".")[:-1]
                    keep = len(package) - (node.level - 1)
                    if keep < 1:
                        raise AssertionError("relative import escapes the scanned package")
                    module = ".".join([*package[:keep], *module.split(".")]).rstrip(".")
                for alias in node.names:
                    local = alias.asname or alias.name
                    self.aliases[local] = f"{module}.{alias.name}"
                    if self.aliases[local] == "functools.partial":
                        self.partial_symbols.add(local)

    def _assignment_defines_pool(self, value: ast.expr) -> bool:
        if self._is_pool_symbol(value):
            return True
        if isinstance(value, ast.Call):
            function = self._resolve_name(value.func)
            return bool(
                function in self.partial_symbols
                and value.args
                and self._is_pool_symbol(value.args[0])
            )
        return False

    def _resolve_pool_symbols(self) -> None:
        changed = True
        while changed:
            changed = False
            for node in ast.walk(self.tree):
                symbol: str | None = None
                is_pool = False
                if isinstance(node, (ast.Assign, ast.AnnAssign)):
                    targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                    value = node.value
                    if value is not None and self._assignment_defines_pool(value):
                        for target in targets:
                            if isinstance(target, ast.Name):
                                symbol = target.id
                                if symbol not in self.pool_symbols:
                                    self.pool_symbols.add(symbol)
                                    changed = True
                elif isinstance(node, ast.ClassDef):
                    symbol = node.name
                    is_pool = any(self._is_pool_symbol(base) for base in node.bases)
                elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    symbol = node.name
                    is_pool = any(
                        isinstance(child, ast.Return)
                        and isinstance(child.value, ast.Call)
                        and self._is_pool_symbol(child.value.func)
                        for child in ast.walk(node)
                    )
                if is_pool and symbol not in self.pool_symbols:
                    self.pool_symbols.add(symbol)
                    changed = True

    def calls(self) -> list[_PoolCall]:
        calls: list[_PoolCall] = []
        owners: list[str] = []
        nonfactory_functions = {
            node.name
            for node in ast.walk(self.tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and not any(
                isinstance(child, ast.Return) and child.value is not None
                for child in ast.walk(node)
            )
        }

        analyzer = self

        class Visitor(ast.NodeVisitor):
            def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
                owners.append(node.name)
                self.generic_visit(node)
                owners.pop()

            visit_AsyncFunctionDef = visit_FunctionDef

            def visit_Call(self, node: ast.Call) -> None:
                if analyzer._is_pool_symbol(node.func):
                    initializer = next(
                        (
                            analyzer._resolve_name(keyword.value) or ast.unparse(keyword.value)
                            for keyword in node.keywords
                            if keyword.arg == "initializer"
                        ),
                        None,
                    )
                    calls.append(_PoolCall(owners[-1] if owners else "<module>", initializer))
                else:
                    name = analyzer._resolve_name(node.func)
                    final_name = name.rsplit(".", 1)[-1] if name else ""
                    lower_name = final_name.lower()
                    pool_shaped = "pool" in lower_name or "executor" in lower_name
                    known_non_process_pool = (
                        final_name in {"ThreadPool", "ThreadPoolExecutor"}
                        or lower_name.endswith(("_size", "_workers", "_class"))
                        or final_name in nonfactory_functions
                        or name in analyzer.package_nonfactory_functions
                    )
                    if (
                        isinstance(node.func, ast.Name)
                        and pool_shaped
                        and not known_non_process_pool
                    ):
                        # Resolution supplies precise inventory entries. Keep a
                        # name backstop as well, because unresolved pool-shaped
                        # factories must fail closed rather than disappear.
                        raise AssertionError(f"unresolved pool-shaped call: {final_name}")
                self.generic_visit(node)

        Visitor().visit(self.tree)
        return calls


@pytest.mark.parametrize(
    "source",
    [
        "from concurrent.futures import ProcessPoolExecutor as PPE\nPPE()",
        (
            "from concurrent.futures import ProcessPoolExecutor\n"
            "def make_pool():\n    return ProcessPoolExecutor()\n"
            "make_pool()\n"
        ),
        (
            "from concurrent.futures import ProcessPoolExecutor\n"
            "class WorkerPool(ProcessPoolExecutor):\n    pass\n"
            "WorkerPool()\n"
        ),
        (
            "from concurrent.futures import ProcessPoolExecutor\n"
            "constructor = ProcessPoolExecutor\nconstructor()\n"
        ),
    ],
    ids=["import-alias", "factory", "subclass", "bound-constructor"],
)
def test_pool_analyzer_finds_indirect_construction_forms(source: str) -> None:
    calls = _PoolAnalyzer(source).calls()
    assert calls
    with pytest.raises(AssertionError):
        assert all(call.initializer is not None for call in calls)


def test_pool_analyzer_finds_partial_constructor() -> None:
    source = (
        "from concurrent.futures import ProcessPoolExecutor\n"
        "from functools import partial\n"
        "WorkerPool = partial(ProcessPoolExecutor, max_workers=2)\n"
        "WorkerPool()\n"
    )
    assert _PoolAnalyzer(source).calls() == [_PoolCall("<module>", None)]


@pytest.mark.parametrize(
    "source",
    [
        "pool_size()\n",
        "_pool_workers()\n",
        "cfg.pool_class()\n",
        "from concurrent.futures import ThreadPoolExecutor\nThreadPoolExecutor()\n",
        "from multiprocessing.pool import ThreadPool\nThreadPool()\n",
    ],
)
def test_non_process_pool_names_are_not_rejected(source: str) -> None:
    assert _PoolAnalyzer(source).calls() == []


def test_unresolved_pool_factory_fails_closed() -> None:
    source = "pool_factory = choose_backend()\npool_factory()\n"
    with pytest.raises(AssertionError, match="unresolved pool-shaped call: pool_factory"):
        _PoolAnalyzer(source).calls()


def _module_name(path: Path) -> str:
    module = ".".join(path.relative_to(ROOT).with_suffix("").parts)
    return module.removesuffix(".__init__")


def _package_pool_symbols(sources: dict[Path, str]) -> set[str]:
    symbols: set[str] = set()
    while True:
        discovered = set(symbols)
        for path, source in sources.items():
            analyzer = _PoolAnalyzer(source, symbols, _module_name(path))
            module_name = _module_name(path)
            for node in analyzer.tree.body:
                if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                    if node.name in analyzer.pool_symbols:
                        discovered.add(f"{module_name}.{node.name}")
                elif isinstance(node, (ast.Assign, ast.AnnAssign)):
                    targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                    for target in targets:
                        if isinstance(target, ast.Name) and target.id in analyzer.pool_symbols:
                            discovered.add(f"{module_name}.{target.id}")
                elif isinstance(node, ast.ImportFrom):
                    for alias in node.names:
                        local = alias.asname or alias.name
                        if analyzer._is_pool_symbol(ast.Name(id=local)):
                            discovered.add(f"{module_name}.{local}")
        if discovered == symbols:
            return symbols
        symbols = discovered


def _pool_inventory(sources: dict[Path, str]) -> dict[tuple[str, str], tuple[str | None, ...]]:
    pools: defaultdict[tuple[str, str], list[str | None]] = defaultdict(list)
    package_pool_symbols = _package_pool_symbols(sources)
    # Apply the existing local nonfactory criterion to definitions in scanned
    # modules, too. Their bodies are still scanned: a constructor inside a
    # context manager remains an inventoried pool, rather than an exemption.
    package_nonfactory_functions = set()
    for path, source in sources.items():
        analyzer = _PoolAnalyzer(source, package_pool_symbols, _module_name(path))
        # A definition describes the exported value only if that name has no
        # other syntactic binding. Count conservatively even inside nested
        # scopes; this gate need not accept ambiguous shadowing.
        bindings: Counter[str] = Counter()
        for binding in ast.walk(analyzer.tree):
            if isinstance(binding, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                bindings[binding.name] += 1
            elif isinstance(binding, ast.Name) and isinstance(binding.ctx, (ast.Store, ast.Del)):
                bindings[binding.id] += 1
            elif isinstance(binding, ast.alias):
                bindings[binding.asname or binding.name.split(".")[0]] += 1
        for node in analyzer.tree.body:
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if bindings[node.name] != 1:
                continue
            # Unknown decorators can replace a function with a pool factory.
            if any(
                analyzer._resolve_name(decorator) != "contextlib.contextmanager"
                for decorator in node.decorator_list
            ):
                continue
            if not any(
                isinstance(child, ast.Return) and child.value is not None
                for child in ast.walk(node)
            ):
                package_nonfactory_functions.add(f"{_module_name(path)}.{node.name}")
    for path, source in sources.items():
        relative_path = path.relative_to(ROOT).as_posix()
        analyzer = _PoolAnalyzer(
            source, package_pool_symbols, _module_name(path), package_nonfactory_functions
        )
        for call in analyzer.calls():
            pools[(relative_path, call.owner)].append(call.initializer)
    return {owner: tuple(initializers) for owner, initializers in pools.items()}


def test_pool_subclass_imported_across_modules_is_inventoryed() -> None:
    sources = {
        ROOT / "pstrain/lib/steps/train.py": (
            "from concurrent.futures import ProcessPoolExecutor\n"
            "class Farm(ProcessPoolExecutor):\n    pass\n"
        ),
        ROOT / "pstrain/lib/pipeline/runner.py": (
            "from pstrain.lib.steps.train import Farm\n"
            "def run():\n    with Farm(max_workers=2):\n        pass\n"
        ),
    }
    assert _pool_inventory(sources) == {("pstrain/lib/pipeline/runner.py", "run"): (None,)}


def test_pool_subclass_imported_relatively_is_inventoryed() -> None:
    sources = {
        ROOT / "pstrain/a.py": (
            "from concurrent.futures import ProcessPoolExecutor\n"
            "class Farm(ProcessPoolExecutor):\n    pass\n"
        ),
        ROOT / "pstrain/b.py": (
            "from .a import Farm\ndef run():\n    with Farm(max_workers=2):\n        pass\n"
        ),
    }
    assert _pool_inventory(sources) == {("pstrain/b.py", "run"): (None,)}


def test_pool_reexports_and_package_initializers_are_inventoryed() -> None:
    sources = {
        ROOT / "pstrain/lib/__init__.py": (
            "from concurrent.futures import ProcessPoolExecutor\n"
            "class Farm(ProcessPoolExecutor):\n    pass\n"
        ),
        ROOT / "pstrain/a.py": (
            "from concurrent.futures import ProcessPoolExecutor\n"
            "class OtherFarm(ProcessPoolExecutor):\n    pass\n"
        ),
        ROOT / "pstrain/c.py": "from .a import OtherFarm\n",
        ROOT / "pstrain/b.py": (
            "from pstrain.lib import Farm\n"
            "from .c import OtherFarm\n"
            "def run():\n    Farm()\n    OtherFarm()\n"
        ),
    }
    assert _pool_inventory(sources) == {
        ("pstrain/b.py", "run"): (None, None),
    }


def test_relative_import_cannot_escape_pstrain() -> None:
    with pytest.raises(AssertionError, match="escapes the scanned package"):
        _PoolAnalyzer("from ...x import Farm\n", module_name="pstrain.lib.b")


def test_runtime_guard_observes_subclass_initializer() -> None:
    source = (
        "from concurrent.futures import ProcessPoolExecutor\n"
        "from pstrain.lib.native_worker import close_helper_before_children_are_joined\n"
        "class SafePool(ProcessPoolExecutor):\n"
        "    def __init__(self, max_workers):\n"
        "        super().__init__(max_workers=max_workers, "
        "initializer=close_helper_before_children_are_joined)\n"
        "pool = SafePool(1)\n"
        "pool.shutdown()\n"
    )
    filename = ROOT / "pstrain/runtime_pool_probe.py"
    exec(compile(source, filename.as_posix(), "exec"), {})


def test_runtime_guard_reports_unsafe_subclass_construction_site() -> None:
    source = (
        "from concurrent.futures import ProcessPoolExecutor\n"
        "class UnsafePool(ProcessPoolExecutor):\n"
        "    pass\n"
        "UnsafePool(1)\n"
    )
    filename = ROOT / "pstrain/runtime_pool_probe.py"
    with pytest.raises(
        AssertionError,
        match=r"constructed at pstrain/runtime_pool_probe\.py:4",
    ):
        exec(compile(source, filename.as_posix(), "exec"), {})


def test_runtime_guard_accepts_thread_pool() -> None:
    source = (
        "from multiprocessing.pool import ThreadPool\n"
        "pool = ThreadPool(1)\n"
        "pool.close()\n"
        "pool.join()\n"
    )
    filename = ROOT / "pstrain/runtime_thread_pool_probe.py"
    exec(compile(source, filename.as_posix(), "exec"), {})


@pytest.mark.parametrize("adapter", ["wraps", "partial"])
def test_runtime_guard_normalizes_approved_initializer_adapters(adapter: str) -> None:
    if adapter == "wraps":
        definition = (
            "@functools.wraps(close_helper_before_children_are_joined)\n"
            "def initializer():\n"
            "    close_helper_before_children_are_joined()\n"
        )
    else:
        definition = "initializer = functools.partial(close_helper_before_children_are_joined)\n"
    source = (
        "import functools\n"
        "from concurrent.futures import ProcessPoolExecutor\n"
        "from pstrain.lib.native_worker import close_helper_before_children_are_joined\n"
        f"{definition}"
        "pool = ProcessPoolExecutor(1, initializer=initializer)\n"
        "pool.shutdown()\n"
    )
    filename = ROOT / "pstrain/runtime_initializer_probe.py"
    exec(compile(source, filename.as_posix(), "exec"), {})


def test_runtime_guard_reports_unhashable_initializer_as_invariant_failure() -> None:
    source = (
        "from concurrent.futures import ProcessPoolExecutor\n"
        "ProcessPoolExecutor(1, initializer=[])\n"
    )
    filename = ROOT / "pstrain/runtime_initializer_probe.py"
    with pytest.raises(
        AssertionError,
        match="initializer must install the native-helper shutdown finalizer",
    ):
        exec(compile(source, filename.as_posix(), "exec"), {})


def _production_pools() -> dict[tuple[str, str], tuple[str | None, ...]]:
    sources = {path: path.read_text(encoding="utf-8") for path in (ROOT / "pstrain").rglob("*.py")}
    return _pool_inventory(sources)


def test_pools_that_can_start_native_helpers_install_the_shutdown_finalizer() -> None:
    # Resolve indirect pool spellings and compare a closed inventory. A newly
    # resolved or initializer-free pool fails instead of becoming an exception.
    expected = POOLS_THAT_INSTALL_HELPER_FINALIZER | POOLS_WITHOUT_NATIVE_HELPERS
    assert _production_pools() == expected


@pytest.mark.parametrize("construct_inside", [False, True])
def test_imported_pool_context_is_not_a_hidden_factory(construct_inside: bool) -> None:
    body = "yield ProcessPoolExecutor()" if construct_inside else "yield existing"
    sources = {
        ROOT / "pstrain/lib/bw_pool.py": (
            "from contextlib import contextmanager\n"
            "from concurrent.futures import ProcessPoolExecutor\n"
            f"@contextmanager\ndef contain_pool(existing):\n    {body}\n"
        ),
        ROOT / "pstrain/lib/steps/train.py": (
            "from pstrain.lib.bw_pool import contain_pool\n"
            "from concurrent.futures import ProcessPoolExecutor\n"
            "def run():\n    with contain_pool(ProcessPoolExecutor()):\n        pass\n"
        ),
    }
    expected = {("pstrain/lib/steps/train.py", "run"): (None,)}
    if construct_inside:
        expected[("pstrain/lib/bw_pool.py", "contain_pool")] = (None,)
    assert _pool_inventory(sources) == expected


def test_unscanned_imported_pool_context_fails_closed() -> None:
    sources = {
        ROOT / "pstrain/lib/steps/train.py": (
            "from external import contain_pool\ncontain_pool(existing)\n"
        )
    }
    with pytest.raises(AssertionError, match="unresolved pool-shaped call: contain_pool"):
        _pool_inventory(sources)


def test_imported_unknown_decorator_cannot_hide_pool_factory() -> None:
    sources = {
        ROOT / "pstrain/lib/bw_pool.py": (
            "from external import replace_with_pool\n"
            "@replace_with_pool\ndef contain_pool(existing):\n    yield existing\n"
        ),
        ROOT / "pstrain/lib/steps/train.py": (
            "from pstrain.lib.bw_pool import contain_pool\ncontain_pool(existing)\n"
        ),
    }
    with pytest.raises(AssertionError, match="unresolved pool-shaped call: contain_pool"):
        _pool_inventory(sources)


@pytest.mark.parametrize(
    "rebind",
    [
        "contain_pool = choose_backend()",
        "contain_pool: object = choose_backend()",
        "contain_pool += choose_backend()",
        "from external import contain_pool",
        "class contain_pool: pass",
        "def contain_pool(): return choose_backend()",
    ],
)
def test_rebound_imported_nonfactory_fails_closed(rebind: str) -> None:
    sources = {
        ROOT / "pstrain/lib/bw_pool.py": (
            "from external import choose_backend\n"
            "def contain_pool(existing):\n    yield existing\n" + rebind + "\n"
        ),
        ROOT / "pstrain/lib/steps/train.py": (
            "from pstrain.lib.bw_pool import contain_pool\ncontain_pool(existing)\n"
        ),
    }
    with pytest.raises(AssertionError, match="unresolved pool-shaped call: contain_pool"):
        _pool_inventory(sources)
