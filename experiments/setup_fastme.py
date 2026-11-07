"""Build the audited FastME release locally; requires a C compiler and autotools."""
from pathlib import Path
import hashlib
import os
import subprocess
import tarfile
import urllib.request

HERE = Path(__file__).resolve().parent
VERSION = "2.1.6.3"
SHA256 = "d49ff78cca7b76eadf8443efb33f071a7e8a152618ffa3d8e790f167d0a0b176"
URL = f"https://gite.lirmm.fr/atgc/FastME/-/archive/v{VERSION}/FastME-v{VERSION}.tar.gz"


def main():
    cache = HERE / ".cache"
    cache.mkdir(exist_ok=True)
    archive = cache / "fastme-source.tar.gz"
    if not archive.exists():
        with urllib.request.urlopen(URL, timeout=120) as response:
            archive.write_bytes(response.read())
    if hashlib.sha256(archive.read_bytes()).hexdigest() != SHA256:
        raise RuntimeError("FastME source checksum mismatch.")
    source = cache / f"FastME-v{VERSION}"
    if not source.exists():
        # The upstream archive includes absolute automake symlinks. Do not
        # extract them; autoreconf regenerates these for the local platform.
        with tarfile.open(archive) as tar:
            members = [m for m in tar.getmembers() if not (m.issym() or m.islnk())]
            tar.extractall(cache, members=members, filter="data")
    prefix = cache / "fastme"
    commands = [["autoreconf", "-fi"], ["./configure", f"--prefix={prefix}"],
                ["make", f"-j{min(4, os.cpu_count() or 1)}"], ["make", "install"]]
    with (cache / "fastme-build.log").open("w") as log:
        for command in commands:
            print(" ".join(command), flush=True)
            subprocess.run(command, cwd=source, stdout=log, stderr=log, check=True)
    subprocess.run([str(prefix / "bin/fastme"), "-V"], check=True)


if __name__ == "__main__":
    main()
