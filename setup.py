from setuptools import setup, find_packages

setup(
    name="snacdb_pipeline",
    version="0.1.0",
    author="Abhinav Gupta",
    author_email="abhinav.gupta@sanofi.com",
    description=(
        "SNAC-DB (Structural NANOBODY® VHH and Antibody (VH-VL) Complex Database) "
        "data curation pipeline"
    ),
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.8, <3.11",
    install_requires=[
        "biopython>=1.85",
        "scipy",
        "tqdm",
        "pytest",
        "pandas",
        "networkx",
    ],
    entry_points={
        "console_scripts": [
            "snacdb-patch-anarci = snacdb.patch:run_patch",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)
