from setuptools import setup, find_packages

with open("./requirements.txt", "r") as f:
    install_reqs = [
        r for r in [l.strip() for l in f.readlines() if l != "\n" and
                    not l.startswith("#")
                    ] 
    ]

setup(
    name="feature_extractor",
    version="1.0.0",
    description= "To be decided",
    author="Gabriel Santos",
    url="https://github.com/gabrielopesantos/feature_extractor",
    install_requires=install_reqs,
    packages=find_packages(exclude=("tests*")),
)
