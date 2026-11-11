"""Fetch the pinned TRL checkout and apply its local import adjustment."""

from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
REVISION = "69ad852e5654a77f1695eb4c608906fe0c7e8624"


def main():
    target = ROOT / "third_party/trl"
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["git", "clone", "https://github.com/huggingface/trl.git", str(target)],
            check=True,
        )
        subprocess.run(
            ["git", "-C", str(target), "checkout", "--detach", REVISION], check=True
        )
    revision = subprocess.check_output(
        ["git", "-C", str(target), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != REVISION:
        raise RuntimeError(f"{target} is at {revision}; expected {REVISION}")
    patch = ROOT / "third_party/patches/trl-transformers5.patch"
    command = ["git", "-C", str(target), "apply"]
    if (
        subprocess.run(
            command + ["--reverse", "--check", str(patch)], capture_output=True
        ).returncode
        == 0
    ):
        print("Pinned TRL and patch already installed")
    else:
        subprocess.run(command + ["--check", str(patch)], check=True)
        subprocess.run(command + [str(patch)], check=True)


if __name__ == "__main__":
    main()
