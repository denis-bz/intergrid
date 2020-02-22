# setup.py (metadata is in setup.cfg)

from setuptools import setup

setup(
    version = "2020.2.20",
    include_package_data = True,  # MANIFEST.in
    long_description_content_type = 'text/markdown',  # https://github.com/pypa/warehouse/issues/5890
    )

