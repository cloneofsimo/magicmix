import os

import pkg_resources

from setuptools import find_packages, setup

setup(
    name="magic_mix",
    py_modules=["magic_mix"],
    version="1.0",
    description="Unofficial Implementation of Magic Mix",
    author="Simo Ryu",
    packages=find_packages(),
    install_requires=[
        str(r)
        for r in pkg_resources.parse_requirements(
            open(os.path.join(os.path.dirname(__file__), "requirements.txt"))
        )
    ],
    include_package_data=True,
)
