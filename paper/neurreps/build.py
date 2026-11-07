"""Build the current manuscript; keep compiler intermediates in build/."""
from pathlib import Path
import shutil
import subprocess
import sys

HERE = Path(__file__).resolve().parent


def main():
    build = HERE / "build"
    build.mkdir(exist_ok=True)
    latex = ["pdflatex", "-interaction=nonstopmode", "-halt-on-error",
             "-output-directory=build", "paper.tex"]
    commands = [[sys.executable, "make_figs.py"], latex, ["bibtex", "build/paper"], latex, latex]
    with (build / "build.log").open("w") as log:
        for command in commands:
            process = subprocess.run(command, cwd=HERE, capture_output=True, text=True)
            log.write("$ " + " ".join(command) + "\n" + process.stdout + process.stderr)
            if process.returncode:
                raise RuntimeError(f"{' '.join(command)} failed. See {build / 'build.log'}\n"
                                   + process.stdout[-3000:] + process.stderr[-1000:])
    text = (build / "paper.log").read_text()
    if any(message in text for message in ("Overfull", "undefined", "Rerun to get cross-references")):
        raise RuntimeError(f"Unresolved LaTeX warnings; see {build / 'paper.log'}")
    shutil.copy2(build / "paper.pdf", HERE / "paper.pdf")
    print(f"Built {HERE / 'paper.pdf'}; no unresolved references or overfull boxes.")


if __name__ == "__main__":
    main()
