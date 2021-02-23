import setuptools

setuptools.setup(
    name="mcx",
    version="0.0.1",
    author="Rémi Louf",
    author_email="remilouf@gmail.com",
    description="Yet Another Probabilistic Programming Library",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/rlouf/mcx",
    licence="Apache Licence Version 2.0",
    packages=setuptools.find_packages(),
    python_requires=">=3.5",
    install_requires=[
        "arviz==0.10.0",
        "libcst",
        "jax==0.2.8",
        "jaxlib==0.1.58",
        "networkx",
        "numpy",
        "tqdm",
    ],
)
