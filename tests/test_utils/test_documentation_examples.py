"""公开文档示例可执行性测试."""

import os
import re
import runpy
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_quickstart_markdown_python_blocks_execute(tmp_path):
    text = (PROJECT_ROOT / "docs" / "quickstart.md").read_text(encoding="utf-8")
    blocks = re.findall(r"```python\n(.*?)```", text, flags=re.S)
    namespace = {"__name__": "__documentation_example__"}
    original_cwd = Path.cwd()
    os.chdir(tmp_path)
    try:
        for index, block in enumerate(blocks, start=1):
            exec(
                compile(block, f"docs/quickstart.md:代码块{index}", "exec"),
                namespace,
            )
    finally:
        os.chdir(original_cwd)
    assert len(blocks) == 8


def test_executable_quickstart_script(tmp_path):
    module = runpy.run_path(str(PROJECT_ROOT / "examples" / "00_quickstart.py"))
    result = module["run_quickstart"](tmp_path)

    assert result["train_rows"] == 300
    assert result["test_rows"] == 100
    assert len(result["scores"]) == 100
    assert Path(result["artifact_path"]).exists()


def test_public_docs_do_not_use_removed_example_parameters():
    paths = [
        PROJECT_ROOT / "README.md",
        PROJECT_ROOT / "docs" / "quickstart.md",
    ]
    text = "\n".join(path.read_text(encoding="utf-8") for path in paths)
    assert "save_path=" not in text
    assert not re.search(r"extract_rules\([^)]*(top_n|metric)", text)
