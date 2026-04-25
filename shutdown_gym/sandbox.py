"""Dictionary-backed simulated filesystem with hash semantics.

The prefix-aware ``_is_readonly`` is the v3 fix: ``readonly_paths``
holds either exact files or directory prefixes, and a path under any
prefix is protected. ``hash`` mixes content + a NUL separator + a
single byte for the executable flag so that ``"abc" + executable``
cannot collide with ``"abcE" + non-executable``.
"""

from dataclasses import dataclass, field
from hashlib import sha256
from typing import Set


@dataclass
class SimFile:
    content: str
    executable: bool = False
    writable: bool = True


@dataclass
class SimulatedFilesystem:
    files: dict[str, SimFile] = field(default_factory=dict)
    readonly_paths: Set[str] = field(default_factory=set)

    def _is_readonly(self, path: str) -> bool:
        """Prefix-aware readonly check.

        ``readonly_paths`` may contain either exact files or directory
        prefixes (with or without a trailing slash). ``"/audit"`` and
        ``"/audit/"`` both protect ``"/audit/commands.log"``.
        """
        for protected in self.readonly_paths:
            protected = protected.rstrip("/")
            if path == protected or path.startswith(protected + "/"):
                return True
        return False

    def read(self, path: str) -> str:
        if path not in self.files:
            raise FileNotFoundError(path)
        return self.files[path].content

    def write(self, path: str, content: str) -> None:
        if self._is_readonly(path):
            raise PermissionError(f"Path is readonly: {path}")
        if path in self.files and not self.files[path].writable:
            raise PermissionError(f"File is not writable: {path}")
        if path in self.files:
            self.files[path].content = content
        else:
            self.files[path] = SimFile(content=content)

    def chmod(self, path: str, executable: bool) -> None:
        if self._is_readonly(path):
            raise PermissionError(f"Path is readonly: {path}")
        if path not in self.files:
            raise FileNotFoundError(path)
        self.files[path].executable = executable

    def delete(self, path: str) -> None:
        if self._is_readonly(path):
            raise PermissionError(f"Path is readonly: {path}")
        if path not in self.files:
            raise FileNotFoundError(path)
        del self.files[path]

    def list(self, directory: str) -> list[str]:
        directory = directory.rstrip("/") + "/"
        return [p for p in self.files if p.startswith(directory)]

    def is_executable(self, path: str) -> bool:
        return path in self.files and self.files[path].executable

    def hash(self, path: str) -> str:
        if path not in self.files:
            return ""
        f = self.files[path]
        h = sha256()
        h.update(f.content.encode())
        h.update(b"\x00")
        h.update(b"E" if f.executable else b"N")
        return h.hexdigest()
