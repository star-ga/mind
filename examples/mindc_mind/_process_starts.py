"""The start rule of the harness, the static half of the one stdin start helper (start_child in _selfhost_sources.py).

Every process start in a tracked Python file is start_child, or it sits in a fn of a PROCESS_STARTS row and passes no
input= or stdin= argument to a child that is not named literally. A row names a fn, not a use: it allows each
reflection in that fn. A start is found through the modules and fns that the tables of this file name (_STARTS,
_EVERY_ATTRIBUTE, _ANY_RECEIVER), bound under any name or from any module, an attribute of another module
(`sources.subprocess.run`), a star import, an import by name, a namespace lookup (sys.modules, vars(), globals(),
locals(), __dict__, a fn's __globals__, a frame's f_globals) or a start module that is passed on as a value. A module
that these tables do not name is not seen, also when it holds a start fn (posix.system is os.system). A binding that
another module could read without the rule seeing it (a start module, sys, builtins or a code module under another
name, a start fn imported by name) is a start.
The rule resolves reflection; it does not refuse each dynamic form by itself. These forms are a start: code from text
(exec, eval, compile, breakpoint, each attribute of a code module of _EVERY_ATTRIBUTE, the loaders and code objects of
_REFLECT); an attribute by text (_REFLECT, getattr as a value, getattr by a name that is no literal and no loop over
literals (_name_values)); a namespace read for its values (each use but a lookup by key, a store or a del by key, an
alias, a membership test and its keys); a ctypes fn of libc in _LIBC (a start or a finder of a fn by name) or a C-API
fn that runs Python (_C_API), on each object that no `import` binds, and a ctypes library by a key that is no literal.
Other dynamic forms pass with no row when the rule resolves their name to no start: __import__, importlib.import_module,
importlib.reload, getattr or a namespace lookup by a literal name or key. setattr has no rule, and a ctypes fn that is
not in _LIBC, _C_API or _ANY_RECEIVER (execveat) is not a start.
A fn with reflection needs a PROCESS_STARTS row that says why it reaches no start.
The rule reads the text of the tracked files: a start inside a module that the tree does not track (a helper of the
standard library, a package) is not seen. start_child's run-time check covers each send through it.
A tracked shell file can not call start_child, so it starts no seed or candidate compiler with stdin bytes.
stdlib_manifest_lint.py runs the rule on the tracked tree and on the plants of PROCESS_START_PLANTS.
"""
import ast
from dataclasses import dataclass, field
import itertools
import math
import pathlib
import re
import subprocess

PROCESS_STARTS = "examples/mindc_mind/testdata/process_starts.tsv"
PROCESS_START_PLANTS = "examples/mindc_mind/testdata/process_start_plants.tsv"
_STARTS = {
    "subprocess": frozenset({"run", "Popen", "call", "check_call", "check_output", "getoutput", "getstatusoutput"}),
    "os": frozenset({"system", "popen", "fork", "forkpty", "posix_spawn", "posix_spawnp", "execl", "execle", "execlp",
                     "execlpe", "execv", "execve", "execvp", "execvpe", "spawnl", "spawnle", "spawnlp", "spawnlpe",
                     "spawnv", "spawnve", "spawnvp", "spawnvpe", "startfile"}),
    "pty": frozenset({"fork", "spawn"}),
    "asyncio": frozenset({"create_subprocess_exec", "create_subprocess_shell"}),
    "_posixsubprocess": frozenset({"fork_exec"}),
}
# Modules whose every attribute is a start: False = it starts a process; True = it runs code from text or bytes
# (reflection).
_EVERY_ATTRIBUTE = {"multiprocessing": False, "webbrowser": False, "runpy": True, "pickle": True, "_pickle": True,
                    "marshal": True, "shelve": True, "code": True, "codeop": True, "zipimport": True, "pdb": True,
                    "bdb": True, "trace": True, "profile": True, "cProfile": True, "timeit": True, "doctest": True,
                    "tkinter": True, "_tkinter": True}
_MODULES = frozenset(_STARTS) | frozenset(_EVERY_ATTRIBUTE)
# Module names that are common attribute words: `X.code` names the module only when an import binds X.
_WORD_MODULES = frozenset({"code", "trace", "profile"})
# These names start a process on any receiver (a module bound to another name, a ctypes libc, an event loop).
_ANY_RECEIVER = frozenset({"Popen", "fork_exec", "forkpty", "posix_spawn", "posix_spawnp", "create_subprocess_exec",
                           "create_subprocess_shell", "subprocess_exec", "subprocess_shell", "ProcessPoolExecutor"}) | {
    n for n in _STARTS["os"] if n.startswith(("exec", "spawn"))}
_START_FNS = frozenset().union(*_STARTS.values()) | _ANY_RECEIVER
# Reflection on any receiver: an attribute or a member by text (operator, inspect, the attribute hooks), a loader or a
# code object (importlib, runpy, types, code), the C API of the interpreter (ctypes), a loader of native code (sqlite3).
_REFLECT = frozenset({"attrgetter", "methodcaller", "getmembers", "getmembers_static", "getattr_static",
                      "__getattribute__", "__getattr__", "spec_from_file_location", "spec_from_loader",
                      "module_from_spec", "exec_module", "load_module", "SourceFileLoader", "SourcelessFileLoader",
                      "ExtensionFileLoader", "run_path", "run_module", "__code__", "FunctionType", "CodeType",
                      "LambdaType", "InteractiveInterpreter", "InteractiveConsole", "compile_command", "pythonapi",
                      "Unpickler", "load_extension", "enable_load_extension"})
_CODE_TEXT = frozenset({"exec", "eval", "compile", "breakpoint"})  # builtins that run text as code
# ctypes: a fn of libc that starts a process or finds another fn by name; a C-API name (_C_API) runs Python.
_LIBC = frozenset({"system", "popen", "fork", "vfork", "clone", "clone3", "wordexp", "syscall", "fexecve", "dlopen",
                   "dlsym", "dlvsym"})
_C_API = re.compile(r"_?Py[A-Z_]")
_LOADERS = frozenset({"CDLL", "PyDLL", "WinDLL", "OleDLL", "LoadLibrary"})  # calls that give a ctypes library
_LIBRARY_LOADERS = frozenset({"cdll", "pydll", "windll", "oledll"})  # `ctypes.cdll.<name>` is a library
_LITERAL_TOOLS = frozenset({"bash", "claude", "git", "sed", "tar"})
_HELPER = ("examples/mindc_mind/_selfhost_sources.py", "start_child")
_IMPORTERS = frozenset({"__import__", "import_module"})  # an import by name gives any module
_NAMESPACE_FNS = frozenset({"vars", "globals", "locals"})
_NAMESPACE_ATTRS = frozenset({"__dict__", "__globals__", "f_globals", "f_locals", "f_builtins"})  # on any receiver
_MODULE_ATTRS = frozenset({"__dict__", "__getattribute__", "__getattr__"})  # a start on a start module
_LOOKUPS = frozenset({"get", "pop", "setdefault"})
_KEY_METHODS = frozenset({"keys", "update"})  # namespace methods that hand out no value
_KEY_FNS = frozenset({"list", "sorted", "set", "frozenset", "tuple", "len", "bool"})  # of a namespace: its keys
# A literal key or attribute name that reaches a namespace, sys, builtins or code by text: a start in any lookup.
_REACHING = frozenset({"sys", "modules", "builtins", "__builtins__", "getattr"}) | (
    _NAMESPACE_FNS | _NAMESPACE_ATTRS | _IMPORTERS | _CODE_TEXT | _REFLECT)
_MAX_VALUES = 64  # the most strings that a name resolution follows
_MAX_DEPTH = 4  # the most names that one resolution follows
_COMPREHENSIONS = (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)
_SCOPES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.Module)
# A seed or candidate compiler in a shell word: a $seed, $stage1, $cand or $compiler variable, or a compiler.elf,
# stage1.elf, cand*.elf or seed*.elf path. The stdin of a shell command: a redirect, a here-document, a here-string.
_SHELL_COMPILER = re.compile(r"\$\{?(?:seed|stage[0-9]|cand(?:idate)?|compiler)\}?(?!\w)"
                             r"|(?:compiler|stage[0-9]|cand\w*|seed\w*)\.elf\b", re.I)
_SHELL_STDIN = re.compile(r"(?<![<&\d])0?<(?![(&>])")
_SHELL_SPLIT = re.compile(r"(\|\||&&|;|\(|\)|\|)")


@dataclass
class _Bindings:
    """The names of one file that hold a start module, a start fn, an import function, sys, builtins, a namespace or a
    ctypes library; `imported` holds the names that an `import` binds (each a module)."""
    modules: dict = field(default_factory=dict)
    names: dict = field(default_factory=dict)
    importers: set = field(default_factory=lambda: set(_IMPORTERS))
    sys_names: set = field(default_factory=set)
    builtins_names: set = field(default_factory=set)
    namespaces: set = field(default_factory=set)
    imported: set = field(default_factory=set)
    libraries: set = field(default_factory=set)


def _module_kind(module: str) -> bool:
    """True when the starts of `module` are reflection (a code module), False for a process start."""
    return _EVERY_ATTRIBUTE.get(module, False)


def _starts_of(module: str) -> frozenset:
    """The starts of a start module; of an unknown module ("?"), every start fn."""
    return _START_FNS if module == "?" else _STARTS.get(module, frozenset())


def _is_start_name(module: str, name: str) -> bool:
    """`module.name` is a start: each attribute of a module of _EVERY_ATTRIBUTE, a start fn of a start module."""
    return module in _EVERY_ATTRIBUTE or name in _starts_of(module)


def _is_c_start(name: str) -> bool:
    return name in _LIBC or bool(_C_API.match(name))


def _from_module(node: ast.ImportFrom) -> str:
    """The start module that `from <module> import` reads ('' for none): the last dotted part that is one."""
    return next((part for part in reversed((node.module or "").split(".")) if part in _MODULES), "")


def _bind_import(found: _Bindings, alias: ast.alias) -> None:
    bound, module = alias.asname or alias.name.split(".")[0], alias.name if alias.asname else alias.name.split(".")[0]
    found.imported.add(bound)
    found.modules.update({bound: module} if module in _MODULES else {})
    found.sys_names.update({bound} if module == "sys" else set())
    found.builtins_names.update({bound} if module == "builtins" else set())


def _bind_from(found: _Bindings, node: ast.ImportFrom, alias: ast.alias) -> None:
    """A star import of a module that is no start module binds every start module, start fn, sys and builtins: it
    may hold any (fail closed)."""
    source, bound = _from_module(node), alias.asname or alias.name
    if alias.name == "*":
        found.names.update({n: (n, False) for n in (_STARTS.get(source, frozenset()) | _ANY_RECEIVER if source
                                                     else _START_FNS)})
        found.modules.update({} if source else {m: m for m in _MODULES})
        found.sys_names.update(set() if source else {"sys"})
        found.builtins_names.update(set() if source else {"builtins"})
    elif alias.name in _MODULES:
        found.modules[bound] = alias.name
    elif source and (_is_start_name(source, alias.name) or alias.name in _ANY_RECEIVER):
        found.names[bound] = (alias.name, _module_kind(source))
    elif alias.name in _IMPORTERS:
        found.importers.add(bound)
    elif alias.name in _CODE_TEXT | _REFLECT:
        found.names[bound] = (alias.name, True)
    elif alias.name in ("sys", "builtins"):
        (found.sys_names if alias.name == "sys" else found.builtins_names).add(bound)
    elif node.module == "sys" and alias.name == "modules":
        found.namespaces.add(bound)


def _assign_pairs(node: ast.Assign):
    """(target, value) of each name an assignment binds; a tuple target pairs with a tuple value of its length."""
    for target in node.targets:
        paired = isinstance(target, ast.Tuple) and isinstance(node.value, ast.Tuple) and len(target.elts) == len(
            node.value.elts)
        yield from zip(target.elts, node.value.elts) if paired else [(target, node.value)]


def _start_bindings(tree: ast.AST) -> _Bindings:
    """The bindings of a file. A start module binds from any module (`from x import subprocess`)."""
    found = _Bindings()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                _bind_import(found, alias)
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                _bind_from(found, node, alias)
    for node in (n for n in ast.walk(tree) if isinstance(n, ast.Assign)):  # `g = globals()` then `g["subprocess"]`
        for target, value in _assign_pairs(node):
            found.namespaces.update({target.id} if isinstance(target, ast.Name) and _is_namespace(value, found) else ())
            name = target.id if isinstance(target, ast.Name) else (
                "." + target.attr if isinstance(target, ast.Attribute) else "")
            found.libraries.update({name} if name and _is_library(value, found) else ())
    return found


def _called_name(node: ast.AST) -> str:
    """The name a call calls (`f(...)` or `x.f(...)`), '' for no call."""
    func = node.func if isinstance(node, ast.Call) else None
    return func.id if isinstance(func, ast.Name) else func.attr if isinstance(func, ast.Attribute) else ""


def _is_call_of(node: ast.AST, name: str) -> bool:
    return _called_name(node) == name


def _literal(node: ast.AST | None) -> str | None:
    return node.value if isinstance(node, ast.Constant) and isinstance(node.value, str) else None


def _is_importer(func: ast.AST, found: _Bindings) -> bool:
    return (isinstance(func, ast.Name) and func.id in found.importers) or (
        isinstance(func, ast.Attribute) and func.attr in _IMPORTERS)


def _imports_name(node: ast.AST, found: _Bindings, name: str) -> bool:
    """An import of `name` by a literal (`importlib.import_module('sys')`)."""
    return isinstance(node, ast.Call) and bool(node.args) and _is_importer(node.func, found) and _literal(
        node.args[0]) == name


def _is_sys(node: ast.AST, found: _Bindings) -> bool:
    """sys: a name bound to it, `X.sys`, an import of it by name."""
    return (isinstance(node, ast.Name) and node.id in found.sys_names) or (
        isinstance(node, ast.Attribute) and node.attr == "sys") or _imports_name(node, found, "sys")


def _is_builtins(node: ast.AST, found: _Bindings) -> bool:
    """builtins: a name bound to it, __builtins__, `X.builtins`, an import of it by name."""
    return (isinstance(node, ast.Name) and (node.id in found.builtins_names or node.id == "__builtins__")) or (
        isinstance(node, ast.Attribute) and node.attr in ("builtins", "__builtins__")) or _imports_name(
        node, found, "builtins")


def _is_library(node: ast.AST, found: _Bindings) -> bool:
    """A ctypes library: a load call (CDLL(...), cdll.LoadLibrary(...)), `ctypes.cdll.<name>`, or a name or an
    attribute that a load was assigned to in this file."""
    if isinstance(node, ast.Call):
        return _called_name(node) in _LOADERS
    if isinstance(node, ast.Attribute):
        receiver = node.value.attr if isinstance(node.value, ast.Attribute) else getattr(node.value, "id", "")
        return "." + node.attr in found.libraries or receiver in _LIBRARY_LOADERS
    return isinstance(node, ast.Name) and node.id in found.libraries


def _is_namespace(node: ast.AST, found: _Bindings) -> bool:
    """vars(), globals(), locals(), x.__dict__, a fn's __globals__, a frame's f_globals, f_locals or f_builtins,
    sys.modules, __builtins__, or a name bound to one of them."""
    if isinstance(node, ast.Call):
        return _called_name(node) in _NAMESPACE_FNS
    if isinstance(node, ast.Attribute):
        return node.attr in _NAMESPACE_ATTRS or (node.attr == "modules" and _is_sys(node.value, found))
    return isinstance(node, ast.Name) and (node.id in found.namespaces or node.id == "__builtins__")


def _lookup_key(node: ast.AST, found: _Bindings) -> ast.AST | None:
    """The key of a namespace lookup (`ns[key]`, `ns.get(key)`), or None when `node` is none."""
    if isinstance(node, ast.Subscript) and _is_namespace(node.value, found):
        return node.slice
    func = node.func if isinstance(node, ast.Call) and node.args else None
    is_lookup = isinstance(func, ast.Attribute) and func.attr in _LOOKUPS and _is_namespace(func.value, found)
    return node.args[0] if is_lookup else None


def _key_starts(key: ast.AST) -> tuple[str, tuple[str, bool] | None]:
    """(start module, start) that a namespace key names. A computed key is a start (fail closed), but `__name__`
    names this module; a key that reaches a namespace, sys, builtins or code by text is reflection."""
    name = _literal(key)
    if name is None:
        return "", None if isinstance(key, ast.Name) and key.id == "__name__" else ("namespace", False)
    head = name.split(".")[0]
    if head in _MODULES:
        return head, None
    return "", (name, False) if name in _START_FNS else (name, True) if name in _REACHING else None


def _module_of(node: ast.AST, found: _Bindings) -> str:
    """The start module that `node` evaluates to ('' for none, "?" for an unknown module: getattr of an imported
    module by a computed name). `X.subprocess` on any receiver is one: another module's import of it, a re-export."""
    if isinstance(node, ast.Name):
        return found.modules.get(node.id, "") if isinstance(node.ctx, ast.Load) else ""
    if isinstance(node, ast.Attribute):
        word = node.attr in _WORD_MODULES and not (
            (isinstance(node.value, ast.Name) and node.value.id in found.imported) or _module_of(node.value, found))
        return node.attr if node.attr in _MODULES and not word else ""
    key = _lookup_key(node, found)
    if key is not None:
        return _key_starts(key)[0]
    if _is_call_of(node, "getattr") and len(node.args) > 1:
        name = _literal(node.args[1])
        if name is None:
            return "?" if isinstance(node.args[0], ast.Name) and node.args[0].id in found.imported else ""
        return name if name in _MODULES else ""
    if isinstance(node, ast.Call) and node.args and _is_importer(node.func, found):
        head = (_literal(node.args[0]) or "").split(".")[0]
        return head if head in _MODULES else ""
    return ""


def _binding_start(node: ast.Import | ast.ImportFrom) -> tuple[str, bool] | None:
    """An import that binds a start where another module reads it unseen: a start module, sys, builtins, a code
    module, an import function or a reflection name under another name, a start fn by name, a star import of a start
    module."""
    if isinstance(node, ast.Import):
        renamed = [a.name for a in node.names if a.asname not in (None, a.name)]
        hit = next((m for m in renamed if m in _MODULES | {"sys", "builtins"}), "")
        return ("import as", hit in ("sys", "builtins") or _module_kind(hit)) if hit else None
    source = _from_module(node)
    for a in node.names:
        renamed = a.asname not in (None, a.name)
        if a.name == "*" and source:
            return "import *", _module_kind(source)
        if renamed and (a.name in _MODULES or a.name in _IMPORTERS):
            return "import as", _module_kind(a.name)
        if renamed and a.name in _REFLECT | {"sys", "builtins", "modules"}:
            return "import as", True
        if (source and _is_start_name(source, a.name)) or a.name in _ANY_RECEIVER:
            return "import " + a.name, _module_kind(source)
        if a.name in _CODE_TEXT:
            return "import " + a.name, True
    return None


def _scope_stores(scope: ast.AST, name: str) -> int:
    """How often `scope` binds `name`: a store, a del, a parameter."""
    stores = sum(isinstance(n, ast.Name) and n.id == name and not isinstance(n.ctx, ast.Load) for n in ast.walk(scope))
    return stores + sum(isinstance(n, ast.arg) and n.arg == name for n in ast.walk(scope))


def _literal_items(node: ast.AST, consts: dict, values: bool = False) -> list | None:
    """The elements of a literal tuple, list or set (or of a module constant bound once to one), the keys of a literal
    dict through .keys() or .items(); with `values`, the values of a literal dict. None for any other node."""
    if isinstance(node, ast.Name) and node.id in consts:
        node = consts[node.id]
    if values and isinstance(node, ast.Dict):
        return list(node.values)
    if isinstance(node, ast.Call) and _called_name(node) in ("keys", "items") and not node.args:
        held = node.func.value
        held = consts.get(held.id, held) if isinstance(held, ast.Name) else held
        if isinstance(held, ast.Dict) and None not in held.keys:
            return [ast.Tuple(elts=[k, v]) for k, v in zip(held.keys, held.values)] if node.func.attr == "items" else (
                list(held.keys))
        return None
    return list(node.elts) if isinstance(node, (ast.Tuple, ast.List, ast.Set)) else None


def _target_values(target: ast.AST, iterable: ast.AST, name: str, consts: dict):
    """(binds, values) of a loop target: does it bind `name`, and the literal strings it takes (None: not literal)."""
    if isinstance(target, ast.Name):
        if target.id != name:
            return False, None
        items = _literal_items(iterable, consts)
        values = [_literal(item) for item in items] if items is not None else [None]
        return True, None if None in values else frozenset(values)
    place = next((i for i, t in enumerate(getattr(target, "elts", [])) if isinstance(t, ast.Name) and t.id == name), None)
    if place is None:
        return False, None
    items = _literal_items(iterable, consts)
    values = [_literal(i.elts[place]) if isinstance(i, ast.Tuple) and len(i.elts) > place else None
              for i in (items if items is not None else [None])]
    return True, None if None in values else frozenset(values)


def _name_values(node: ast.AST, scope: tuple, depth: int = 0) -> frozenset | None:
    """The strings that `node` may hold: a literal, a sum, an f-string or a conditional of such parts, a lookup in a
    literal dict, tuple or list, or a name: the target of the nearest for loop or comprehension over literals (_literal_items) that
    binds it, bound nowhere else in that loop, or a name that its fn binds once, to such a value. A module constant
    counts only at module level: in a fn, another module may have bound it again. None for any other node (and past
    _MAX_DEPTH names)."""
    if _literal(node) is not None:
        return frozenset({node.value})
    parent, consts = scope
    home = node
    while home in parent and not isinstance(home, _SCOPES):
        home = parent[home]
    consts = consts if isinstance(home, ast.Module) else {}
    if depth > _MAX_DEPTH or not isinstance(node, (ast.BinOp, ast.JoinedStr, ast.Subscript, ast.Name, ast.IfExp)):
        return None
    if isinstance(node, ast.IfExp):
        branches = [_name_values(node.body, scope, depth + 1), _name_values(node.orelse, scope, depth + 1)]
        return None if None in branches else branches[0] | branches[1]
    if not isinstance(node, (ast.Subscript, ast.Name)):
        return _joined_values(node, scope, depth)
    if isinstance(node, ast.Subscript):
        items = _literal_items(node.value, consts, values=True)
        found = [_literal(item) for item in items] if items else [None]
        return None if None in found else frozenset(found)
    up = node
    while up is not home:
        up = parent[up]
        loops = [(up, up.target, up.iter)] if isinstance(up, (ast.For, ast.AsyncFor)) else [
            (up, g.target, g.iter) for g in up.generators] if isinstance(up, _COMPREHENSIONS) else []
        for loop, target, iterable in loops:
            binds, values = _target_values(target, iterable, node.id, consts)
            if binds:
                return values if _scope_stores(loop, node.id) == 1 else None
    bound = [n.value for n in ast.walk(home) if isinstance(n, ast.Assign) and len(n.targets) == 1
             and isinstance(n.targets[0], ast.Name) and n.targets[0].id == node.id]
    once = len(bound) == 1 and _scope_stores(home, node.id) == 1
    return _name_values(bound[0], scope, depth + 1) if once else None


def _joined_values(node: ast.AST, scope: tuple, depth: int) -> frozenset | None:
    """The strings of `a + b` or of an f-string whose parts each hold known strings (at most _MAX_VALUES)."""
    if isinstance(node, ast.BinOp):
        parts = [_name_values(node.left, scope, depth), _name_values(node.right, scope, depth)] if isinstance(
            node.op, ast.Add) else [None]
    else:
        parts = [frozenset({v.value}) if isinstance(v, ast.Constant) else _name_values(v.value, scope, depth) if (
            isinstance(v, ast.FormattedValue) and v.conversion == -1 and v.format_spec is None) else None
            for v in node.values]
    if None in parts or math.prod(len(p) for p in parts) > _MAX_VALUES:
        return None
    return frozenset("".join(words) for words in itertools.product(*parts))


def _getattr_start(node: ast.Call, found: _Bindings, scope: tuple) -> tuple[str, bool] | None:
    """getattr on a start module for a start or a computed name; on any receiver for a start of _ANY_RECEIVER or a
    name of _REACHING (reflection); for a ctypes start on an object that no import binds; and by a name that is no
    literal and no loop over literals (reflection)."""
    module = _module_of(node.args[0], found)
    names = _name_values(node.args[1], scope) if len(node.args) > 1 else None
    if names is None:
        return ("getattr", _module_kind(module)) if module else ("getattr by name", True)
    receiver_imported = isinstance(node.args[0], ast.Name) and node.args[0].id in found.imported
    for name in sorted(names):
        if (module and (_is_start_name(module, name) or name in _MODULE_ATTRS)) or name in _ANY_RECEIVER:
            return "getattr", _module_kind(module)
        if name in _MODULES:
            return "getattr", _module_kind(name)
        if name in _REACHING or (_is_c_start(name) and not receiver_imported):
            return "getattr", True
    return None


def _attribute_start(node: ast.Attribute, found: _Bindings, called: bool) -> tuple[str, bool] | None:
    """An attribute that starts: on any receiver (_ANY_RECEIVER, _REFLECT), on a start module, exec/eval/compile of
    builtins, a ctypes start on an object that no import binds, an import function used as a value."""
    module = _module_of(node.value, found)
    if node.attr in _ANY_RECEIVER:
        return node.attr, False
    if module and (_is_start_name(module, node.attr) or node.attr in _MODULE_ATTRS):
        return node.attr, _module_kind(module)
    if node.attr in _REFLECT or (node.attr in _CODE_TEXT and _is_builtins(node.value, found)):
        return node.attr, True
    if _is_c_start(node.attr) and not (isinstance(node.value, ast.Name) and node.value.id in found.imported):
        return node.attr, True
    return ("import", False) if node.attr in _IMPORTERS and not called else None


def _name_start(node: ast.Name, found: _Bindings, called: bool) -> tuple[str, bool] | None:
    """A name that starts: a bound start fn, exec/eval/compile/breakpoint or a reflection name, getattr or a namespace
    fn used as a value, an import function used as a value."""
    if not isinstance(node.ctx, ast.Load):
        return None
    if node.id in found.names:
        return found.names[node.id]
    if node.id in _CODE_TEXT or node.id in _REFLECT:
        return node.id, True
    if not called and node.id in _NAMESPACE_FNS | {"getattr"}:
        return node.id + " as a value", True
    return ("import", False) if node.id in found.importers and not called else None


def _call_start(node: ast.Call, found: _Bindings, scope: tuple) -> tuple[str, bool] | None:
    """A call that starts: a namespace lookup by key, an import by a computed name, getattr (_getattr_start), a ctypes
    prototype bound to a libc start by name (`proto(("system", libc))`)."""
    key = _lookup_key(node, found)
    if key is not None:
        return _key_starts(key)[1]
    if node.args and _is_importer(node.func, found) and _literal(node.args[0]) is None:
        return "import", False
    if _is_call_of(node, "getattr") and node.args:
        return _getattr_start(node, found, scope)
    named = [_literal(a.elts[0]) for a in node.args if isinstance(a, ast.Tuple) and a.elts]
    return next(((name, True) for name in named if name and _is_c_start(name)), None)


def _subscript_start(node: ast.Subscript, found: _Bindings, scope: tuple) -> tuple[str, bool] | None:
    """A namespace lookup by key; a ctypes start by a literal key (`libc["system"]`), a ctypes library by a key that
    is no literal and no loop over literals (reflection)."""
    key = _lookup_key(node, found)
    if key is not None:
        return _key_starts(key)[1]
    name = _literal(node.slice)
    if name is not None:
        return (name, True) if _is_c_start(name) else None
    if not _is_library(node.value, found):
        return None
    names = _name_values(node.slice, scope)
    hit = next((n for n in sorted(names or ()) if _is_c_start(n)), None)
    return ("library by key", True) if names is None else (hit, True) if hit else None


def _is_value_read(node: ast.AST, parent: dict) -> bool:
    """A namespace read for its values: each use but a lookup by key, a store or a del by key, an alias, a membership
    test, a loop over its keys, a key fn (_KEY_FNS) or a key method (_KEY_METHODS)."""
    up = parent.get(node)
    if isinstance(up, ast.Subscript) and up.value is node:
        return False
    if isinstance(up, ast.Attribute) and up.value is node:
        call = parent.get(up)
        is_lookup = up.attr in _LOOKUPS and isinstance(call, ast.Call) and call.func is up and bool(call.args)
        return not (is_lookup or up.attr in _KEY_METHODS)
    if isinstance(up, ast.Assign) and up.value is node:
        return not all(isinstance(t, ast.Name) for t in up.targets)
    if isinstance(up, ast.Tuple) and isinstance(parent.get(up), ast.Assign):
        return not any(value is node and isinstance(target, ast.Name) for target, value in _assign_pairs(parent[up]))
    if isinstance(up, ast.Compare):
        return not (any(c is node for c in up.comparators) and all(isinstance(o, (ast.In, ast.NotIn)) for o in up.ops))
    if isinstance(up, (ast.For, ast.AsyncFor, ast.comprehension)) and up.iter is node:
        return False
    return not (isinstance(up, ast.Call) and _called_name(up) in _KEY_FNS and up.args == [node] and not up.keywords
                and isinstance(up.func, ast.Name))


def _start_of(node: ast.AST, found: _Bindings, scope: tuple) -> tuple[str, bool] | None:
    """The start that `node` makes, (label, reflection), or None. `scope` = (parents, module constants)."""
    parent = scope[0]
    up = parent.get(node)
    if isinstance(node, (ast.Import, ast.ImportFrom)):
        return _binding_start(node)
    called = isinstance(up, ast.Call) and up.func is node
    start = (_attribute_start(node, found, called) if isinstance(node, ast.Attribute) else
             _name_start(node, found, called) if isinstance(node, ast.Name) else
             _call_start(node, found, scope) if isinstance(node, ast.Call) else
             _subscript_start(node, found, scope) if isinstance(node, ast.Subscript) else None)
    if start:
        return start
    loaded = isinstance(node, ast.Call) or isinstance(getattr(node, "ctx", None), ast.Load)
    if loaded and _is_namespace(node, found) and _is_value_read(node, parent):
        return "namespace read", True
    return ("module", False) if _module_of(node, found) not in ("", "?") and not _is_read(node, up) else None


def _is_read(node: ast.AST, up: ast.AST | None) -> bool:
    """A start module read for an attribute, or by getattr or hasattr: the reader is judged, not the module."""
    return (isinstance(up, ast.Attribute) and up.value is node) or (
        (_is_call_of(up, "getattr") or _is_call_of(up, "hasattr")) and bool(up.args) and up.args[0] is node)


def _start_refs(tree: ast.AST) -> list[tuple[ast.AST, tuple[str, bool], str, ast.Call | None]]:
    """(node, (start, reflection), enclosing fn, the call that runs it or None) of each start of a file. A type
    annotation and the class argument of isinstance/issubclass start nothing."""
    found = _start_bindings(tree)
    parent = {child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}
    consts = {t.id: n.value for n in getattr(tree, "body", []) if isinstance(n, ast.Assign) for t in n.targets
              if isinstance(t, ast.Name) and _scope_stores(tree, t.id) == 1}
    skip = {id(n) for node in ast.walk(tree) for ann in (
        [getattr(node, "annotation", None), getattr(node, "returns", None)]
        + (node.args[1:] if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
           and node.func.id in ("isinstance", "issubclass") else []))
        if ann is not None for n in ast.walk(ann)}
    refs = []
    for node in ast.walk(tree):
        start = None if id(node) in skip else _start_of(node, found, (parent, consts))
        if not start:
            continue
        fn, up = "<module>", node
        while up in parent and fn == "<module>":
            up = parent[up]
            fn = up.name if isinstance(up, (ast.FunctionDef, ast.AsyncFunctionDef)) else fn
        call = parent.get(node)
        refs.append((node, start, fn, call if isinstance(call, ast.Call) and call.func is node else None))
    return refs


def _stdin_bytes(call: ast.Call, start: str) -> bool:
    """True when the call may hand bytes to the child's stdin: input=, stdin= other than DEVNULL or None, **kwargs, a
    positional stdin, os.popen for writing, posix_spawn file actions."""
    kws = {k.arg: k.value for k in call.keywords}
    stdin = kws.get("stdin")
    if None in kws or "input" in kws or "file_actions" in kws or (start == "popen" and (len(call.args) > 1 or "mode" in kws)):
        return True
    if start in _STARTS["subprocess"] and len(call.args) > 1:
        return True
    return stdin is not None and not (isinstance(stdin, ast.Attribute) and stdin.attr == "DEVNULL") and not (
        isinstance(stdin, ast.Constant) and stdin.value is None)


def _literal_tool(call: ast.Call) -> bool:
    """The child is named literally: a tool of _LITERAL_TOOLS or sys.executable (never a seed or candidate path)."""
    argv = call.args[0] if call.args else None
    head = argv.elts[0] if isinstance(argv, (ast.List, ast.Tuple)) and argv.elts else None
    return (isinstance(head, ast.Constant) and head.value in _LITERAL_TOOLS) or (
        isinstance(head, ast.Attribute) and head.attr == "executable" and isinstance(head.value, ast.Name)
        and head.value.id == "sys")


def _start_table(path: pathlib.Path) -> tuple[dict[tuple[str, str], str], list[str]]:
    """PROCESS_STARTS: a header, then one row per (file, fn, reason)."""
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        return {}, [f"process starts: the table {path.name} can not be read ({error})"]
    rows, errors = {}, [] if lines[:1] == ["file\tfunction\treason"] else [f"{path.name}: the header is not file, function, reason"]
    for number, line in enumerate(lines[1:], 2):
        fields = line.split("\t")
        if len(fields) != 3 or not all(f.strip() for f in fields) or (fields[0], fields[1]) in rows:
            errors.append(f"{path.name}:{number}: a row needs a file, a fn and a reason, once: {line[:120]!r}")
            continue
        rows[(fields[0], fields[1])] = fields[2]
    return rows, errors


def shell_start_errors(rel: str, text: str) -> list[str]:
    """Each command of a shell text that starts a seed or candidate compiler with stdin bytes (a redirect, a
    here-document or a here-string after the compiler word, or a pipe into it). A continued line is one line."""
    errors, number = [], 1
    for line in re.split(r"(?<!\\)\n", text):
        here, number = number, number + line.count("\n") + 1
        parts = [] if line.lstrip().startswith("#") else _SHELL_SPLIT.split(line.replace("\\\n", " "))
        for index in range(0, len(parts), 2):
            segment, redirect = parts[index], _SHELL_STDIN.search(parts[index])
            head = segment if index and parts[index - 1] == "|" else segment[:redirect.start()] if redirect else ""
            if _SHELL_COMPILER.search(head):
                errors.append(f"{rel}:{here}: a seed or candidate started by a shell with stdin bytes; "
                              "a shell can not call start_child: build it from Python through start_child")
    return errors


def _python_start_errors(rel: str, tree: ast.AST, rows: dict, seen: set, table: str) -> list[str]:
    """The starts of one Python file that no row allows. A row allows a process start that is called and hands no
    stdin bytes to a child that is not named literally, and reflection in a fn that starts no process: beside a
    start, reflection could hand the child bytes that the rule can not see."""
    errors, refs = [], [ref for ref in _start_refs(tree) if (rel, ref[2]) != _HELPER]
    starting = {fn for _, (_, reflection), fn, _ in refs if not reflection}
    for node, (start, reflection), fn, call in refs:
        where = f"{rel}:{node.lineno}: {start} in {fn}()"
        if (rel, fn) not in rows:
            kind = "reflection (the rule can not see what it reaches)" if reflection else "a process start"
            errors.append(f"{where}: {kind} outside start_child and outside {table}")
            continue
        seen.add((rel, fn))
        if reflection:
            errors += [f"{where}: reflection in a fn that also starts a process; put it in a fn of its own"] if (
                fn in starting) else []
            continue
        if call is None:
            errors.append(f"{where}: a process start used as a value; call it, or start the child through start_child")
        elif _stdin_bytes(call, start) and not _literal_tool(call):
            errors.append(f"{where}: stdin bytes for a child not named literally; start it through start_child")
    return errors


def process_start_errors(root: pathlib.Path, table: str = PROCESS_STARTS) -> list[str]:
    """Every process start in a tracked Python file is start_child, or sits in a fn of a `table` row and hands no stdin
    bytes to a child that is not named literally; no tracked shell file starts a seed or candidate with stdin bytes.
    A row with no start in its fn is an error, so the table stays exact."""
    root = pathlib.Path(root)
    listed = subprocess.run(["git", "ls-files", "-z", "--", "*.py", "*.sh"], cwd=root, capture_output=True)
    if listed.returncode != 0:
        return [f"process starts: git ls-files failed: {listed.stderr.decode(errors='replace')[:200]}"]
    rows, errors = _start_table(root / table)
    seen = set()
    for rel in sorted(p for p in listed.stdout.decode().split("\0") if p.endswith((".py", ".sh"))):
        path = root / rel
        if path.is_symlink() or not path.is_file():
            continue
        if rel.endswith(".sh"):
            errors += shell_start_errors(rel, path.read_text(encoding="utf-8", errors="replace"))
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=rel)
        except SyntaxError as exc:
            errors.append(f"{rel}: cannot parse for process starts ({exc})")
            continue
        errors += _python_start_errors(rel, tree, rows, seen, table)
    return errors + [f"{table}: the row {f} {fn} names no process start" for f, fn in sorted(set(rows) - seen)]
