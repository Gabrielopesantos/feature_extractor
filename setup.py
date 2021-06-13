import setuptools

with open("./requirements.txt", "r") as f:
    install_reqs = [
        r for r in [l.strip() for l in f.readlines() if l != "\n" and
                    not l.startswith("#")
                    ] 
    ]

setuptools.setup(
    name="feature_extractor",
    version="0.0.1",
    packages=["feature_extractor"],
    install_requires=install_reqs,
)
