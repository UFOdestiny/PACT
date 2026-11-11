"""Code answers: extraction from a response and correctness by sandboxed execution against the set's tests.

A code set row carries ``code_prefix`` (the prompt's own definitions and imports that the tests may rely on) and
``code_tests`` (the dataset's test program, ending in the call that runs it), so the judged program is
``code_prefix + answer + code_tests``.  It runs in a fresh ``python -I`` subprocess in a temporary directory, never
in-process.  Inside the child ``resource.setrlimit`` caps CPU time (30 s per program, i.e. all of one problem's
tests), address space (4 GiB), file size and core dumps; the network is cut (a fresh network namespace via
``unshare -rn`` when the host allows it, and the socket module disabled either way); process creation and destructive
``os`` / ``subprocess`` / ``shutil`` calls are removed; stdin is empty; and the process group is killed at a 90 s
wall-clock timeout.  The answer passes only when the program exits 0 after printing the runner's completion line, so
an early ``sys.exit(0)`` is a failure.
"""

from __future__ import annotations

import os
import re
import signal
import subprocess
import sys
import tempfile
from pathlib import Path

CPU_SECONDS = 30                  # per judged program (all of one problem's tests); the slowest reference takes ~11 s
WALL_SECONDS = 90.0               # wall clock, looser than the CPU limit so a busy node does not fail a correct answer
MEMORY_BYTES = 4 * 1024 ** 3      # address space
FILE_BYTES = 64 * 1024 ** 2       # largest file the program may write
_DONE = "__PACT_TESTS_COMPLETED__"

_FENCE = re.compile(r"```[ \t]*([A-Za-z0-9_+-]*)[ \t]*\n(.*?)```", re.DOTALL)
_ANSWER = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)


def extract_code(text: str) -> str | None:
    """The answer's Python program, or None.

    Searched in the last ``<answer>`` block when the response has one (tag protocols), else in the whole response:
    the last fenced block labelled python/py (or unlabelled) that defines a function or imports, else the last such
    block; an ``<answer>`` block without a fence is read as code.  An unterminated fence (a truncated response) is no
    answer.
    """
    answers = _ANSWER.findall(text)
    scope = answers[-1] if answers else text
    blocks = [body for label, body in _FENCE.findall(scope) if label.lower() in ("", "python", "py", "python3")]
    if blocks:
        defining = [b for b in blocks if re.search(r"(?m)^\s*(?:def |class |import |from \S+ import )", b)]
        code = (defining or blocks)[-1]
    elif answers:
        code = answers[-1]
    else:
        return None
    return code.strip("\n") if code.strip() else None


# The child's preamble: limits, network and destructive calls off, then the program under test.
_RUNNER = r"""
import builtins, os, resource, sys
cpu, memory, files = int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
for limit, value in ((resource.RLIMIT_CPU, cpu), (resource.RLIMIT_AS, memory),
                     (resource.RLIMIT_FSIZE, files), (resource.RLIMIT_CORE, 0)):
    resource.setrlimit(limit, (value, value))
import faulthandler, math, numpy  # imported before the guard, the tests use numpy
import socket
def _no_network(*args, **kwargs):
    raise OSError("network access is disabled")
for name in ("socket", "create_connection", "create_server", "socketpair", "fromfd", "getaddrinfo"):
    setattr(socket, name, _no_network)
import shutil, subprocess
for module, names in ((os, ("system", "kill", "killpg", "fork", "forkpty", "execv", "execve", "execvp", "execvpe",
                            "spawnv", "spawnve", "popen", "putenv", "remove", "unlink", "rmdir", "removedirs",
                            "rename", "renames", "replace", "truncate", "chmod", "chown", "setuid", "_exit")),
                      (subprocess, ("Popen", "run", "call", "check_call", "check_output", "getoutput")),
                      (shutil, ("rmtree", "move", "chown", "copyfile", "copytree"))):
    for name in names:
        if hasattr(module, name):
            setattr(module, name, None)
builtins.exit = builtins.quit = None
source = open(sys.argv[1], encoding="utf-8").read()
namespace = {"__name__": sys.argv[5], "__builtins__": builtins}
try:
    exec(compile(source, "program.py", "exec"), namespace)
except BaseException as error:
    sys.stderr.write(f"{type(error).__name__}: {error}\n"[:2000])
    sys.stderr.flush()
    sys.stdout.flush()
    sys.exit(1)
sys.stdout.write("\n" + sys.argv[6] + "\n")
sys.stdout.flush()
"""

_UNSHARE: list[bool] = []    # probed once per process: does ``unshare -rn`` work here?


def _network_prefix() -> list[str]:
    if not _UNSHARE:
        try:
            ok = subprocess.run(["unshare", "-rn", "true"], capture_output=True, timeout=10).returncode == 0
        except (OSError, subprocess.SubprocessError):
            ok = False
        _UNSHARE.append(ok)
    return ["unshare", "-rn"] if _UNSHARE[0] else []


def run_program(source: str, *, cpu_seconds: int = CPU_SECONDS, timeout: float = WALL_SECONDS,
                module_name: str = "__candidate__",
                stdin: str | None = None) -> tuple[bool, str]:
    """Execute ``source`` in the sandbox; ``(passed, stage)`` with stage ``exec`` (passed), ``reject`` (an error or
    failed assertion, or the CPU limit), or ``timeout`` (wall clock)."""
    with tempfile.TemporaryDirectory(prefix="pact-code-") as work:
        program = Path(work) / "program.py"
        program.write_text(source, encoding="utf-8")
        env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "HOME": work, "TMPDIR": work, "LANG": "C.UTF-8",
               "PYTHONDONTWRITEBYTECODE": "1", "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
               "MKL_NUM_THREADS": "1"}
        command = [*_network_prefix(), sys.executable, "-I", "-c", _RUNNER, str(program), str(cpu_seconds),
                   str(MEMORY_BYTES), str(FILE_BYTES), module_name, _DONE]
        # stdout goes to a file in the work directory (bounded by the file-size limit), not into this process
        with open(Path(work) / "stdout.txt", "wb") as out:
            process = subprocess.Popen(command, cwd=work, env=env,
                                       stdin=subprocess.PIPE if stdin is not None else subprocess.DEVNULL,
                                       stdout=out, stderr=subprocess.DEVNULL, start_new_session=True)
            timed_out = False
            try:
                process.communicate(stdin.encode() if stdin is not None else None, timeout=timeout)
            except subprocess.TimeoutExpired:
                timed_out = True
            finally:
                if process.poll() is None:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)   # the program's whole process group
                    except (ProcessLookupError, PermissionError):
                        pass
                    process.wait()
        if timed_out:
            return False, "timeout"
        with open(Path(work) / "stdout.txt", "rb") as handle:
            handle.seek(max(0, os.path.getsize(handle.name) - 4096))
            tail = handle.read().decode("utf-8", "replace")
    passed = process.returncode == 0 and tail.rstrip().endswith(_DONE)
    return passed, "exec" if passed else "reject"


def judge(code: str | None, row: dict) -> tuple[bool, str]:
    """Correctness of an extracted answer against one code row's tests (see the module docstring)."""
    if code is None:
        return False, "unparsed"
    source = f"{row['code_prefix']}\n\n{code}\n\n{row['code_tests']}\n"
    return run_program(source)
