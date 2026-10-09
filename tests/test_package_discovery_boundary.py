"""发布包只能包含库本身，不能自动发现outputs中的第三方或用户代码。"""
from pathlib import Path
import sys
from setuptools import find_namespace_packages

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib


def test_package_discovery_is_explicitly_scoped_to_hscredit(tmp_path):
    root = Path(__file__).resolve().parents[1]
    config = tomllib.loads((root / 'pyproject.toml').read_text(encoding='utf-8'))
    options = config['tool']['setuptools']['packages']['find']
    for name in ['hscredit/core', 'outputs/customer_private', 'outputs/node_modules/tool', '.audit_tmp/build']:
        (tmp_path / name).mkdir(parents=True)
    packages = find_namespace_packages(where=str(tmp_path), include=options['include'], exclude=options['exclude'])
    assert 'hscredit' in packages and 'hscredit.core' in packages
    assert all(name == 'hscredit' or name.startswith('hscredit.') for name in packages)
