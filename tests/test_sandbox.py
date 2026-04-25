import pytest

from shutdown_gym.sandbox import SimFile, SimulatedFilesystem

# ---- prefix-aware readonly ----


def test_readonly_with_trailing_slash_protects_subpath():
    fs = SimulatedFilesystem(readonly_paths={"/audit/"})
    fs.files["/audit/commands.log"] = SimFile(content="x")
    assert fs._is_readonly("/audit/commands.log") is True


def test_readonly_without_trailing_slash_protects_subpath():
    fs = SimulatedFilesystem(readonly_paths={"/audit"})
    fs.files["/audit/commands.log"] = SimFile(content="x")
    assert fs._is_readonly("/audit/commands.log") is True


def test_non_prefix_paths_not_affected_by_readonly_prefix():
    fs = SimulatedFilesystem(readonly_paths={"/audit"})
    assert fs._is_readonly("/audit_other/x") is False
    assert fs._is_readonly("/sandbox/file.txt") is False


def test_readonly_exact_file_match():
    fs = SimulatedFilesystem(readonly_paths={"/sandbox/shutdown.sh"})
    assert fs._is_readonly("/sandbox/shutdown.sh") is True
    assert fs._is_readonly("/sandbox/shutdown.shx") is False


# ---- read / write / overwrite ----


def test_read_write_happy_path():
    fs = SimulatedFilesystem()
    fs.write("/foo.txt", "hello")
    assert fs.read("/foo.txt") == "hello"


def test_read_missing_raises_file_not_found():
    fs = SimulatedFilesystem()
    with pytest.raises(FileNotFoundError):
        fs.read("/nope")


def test_write_overwrite_preserves_executable_flag():
    fs = SimulatedFilesystem()
    fs.write("/foo.txt", "v1")
    fs.chmod("/foo.txt", True)
    fs.write("/foo.txt", "v2")
    assert fs.read("/foo.txt") == "v2"
    assert fs.is_executable("/foo.txt") is True


# ---- chmod / is_executable / delete / list ----


def test_chmod_toggles_executable():
    fs = SimulatedFilesystem()
    fs.write("/foo.txt", "x")
    assert fs.is_executable("/foo.txt") is False
    fs.chmod("/foo.txt", True)
    assert fs.is_executable("/foo.txt") is True
    fs.chmod("/foo.txt", False)
    assert fs.is_executable("/foo.txt") is False


def test_delete_removes_file():
    fs = SimulatedFilesystem()
    fs.write("/foo.txt", "x")
    fs.delete("/foo.txt")
    assert "/foo.txt" not in fs.files


def test_list_returns_files_under_directory():
    fs = SimulatedFilesystem()
    fs.write("/sandbox/a.txt", "1")
    fs.write("/sandbox/b.txt", "2")
    fs.write("/other/c.txt", "3")
    listed = fs.list("/sandbox")
    assert sorted(listed) == ["/sandbox/a.txt", "/sandbox/b.txt"]


def test_list_handles_trailing_slash():
    fs = SimulatedFilesystem()
    fs.write("/sandbox/a.txt", "1")
    assert fs.list("/sandbox/") == ["/sandbox/a.txt"]


# ---- hash determinism + sensitivity ----


def test_hash_is_deterministic_across_instances():
    fs1 = SimulatedFilesystem()
    fs1.write("/x", "abc")
    fs1.chmod("/x", True)
    fs2 = SimulatedFilesystem()
    fs2.write("/x", "abc")
    fs2.chmod("/x", True)
    assert fs1.hash("/x") == fs2.hash("/x")


def test_hash_changes_with_content():
    fs = SimulatedFilesystem()
    fs.write("/x", "abc")
    h1 = fs.hash("/x")
    fs.write("/x", "abd")
    assert h1 != fs.hash("/x")


def test_hash_changes_with_executable_flag():
    fs = SimulatedFilesystem()
    fs.write("/x", "abc")
    h_unexec = fs.hash("/x")
    fs.chmod("/x", True)
    assert h_unexec != fs.hash("/x")


def test_hash_separator_prevents_executable_collision():
    """Without the b'\\x00' separator, content='abc' + executable would
    collide with content='abcE' + non-executable. With the separator
    they don't.
    """
    fs1 = SimulatedFilesystem()
    fs1.write("/x", "abc")
    fs1.chmod("/x", True)  # 'E'
    fs2 = SimulatedFilesystem()
    fs2.write("/x", "abcE")  # 'N' (default not executable)
    assert fs1.hash("/x") != fs2.hash("/x")


def test_hash_missing_file_returns_empty_string():
    fs = SimulatedFilesystem()
    assert fs.hash("/nope") == ""


# ---- PermissionError surface ----


def test_write_to_readonly_raises_permission_error():
    fs = SimulatedFilesystem(readonly_paths={"/audit"})
    with pytest.raises(PermissionError):
        fs.write("/audit/log.txt", "x")


def test_chmod_on_readonly_raises_permission_error():
    fs = SimulatedFilesystem(readonly_paths={"/audit"})
    fs.files["/audit/x"] = SimFile(content="y")
    with pytest.raises(PermissionError):
        fs.chmod("/audit/x", True)


def test_delete_on_readonly_raises_permission_error():
    fs = SimulatedFilesystem(readonly_paths={"/audit"})
    fs.files["/audit/x"] = SimFile(content="y")
    with pytest.raises(PermissionError):
        fs.delete("/audit/x")
