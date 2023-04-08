import os

from setuptools import find_packages, setup


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname), encoding="UTF-8").read()


setup(
    name="spvqe",
    version="0.0.1",
    description="Package to study the Series of Penalty VQE variation",
    long_description=read("README.MD"),
    author="Rodolfo Carobene",
    author_email="r.carobene@campus.unimib.it",
    url="https://github.com/rodolfocarobene/SPVQE",
    license=read("LICENSE"),
    install_requires=read("requirements.txt").splitlines(),
    package_dir={"": "."},
    packages=find_packages(where=".", exclude=("notebooks")),
    python_requires=">=3.8",
    zip_safe=False,
)
