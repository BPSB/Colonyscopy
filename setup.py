
from setuptools import setup

requirements = ["numpy","scipy","matplotlib","Pillow","psutil","rawpy","ipywidgets","IPython"]
setup(
    name="colonyscopy",
    description="Analysis of colony growth over time.",
    long_description=open("README.md", encoding="utf8").read(),
    long_description_content_type="text/markdown",
    python_requires=">=3.10",
    packages=["colonyscopy"],
    install_requires=requirements,
    use_scm_version={"write_to": "colonyscopy/version.py"},
    classifiers=[
        "Development Status :: Alpha",
        "License :: OSI Approved :: BSD License",
        "Operating System :: POSIX",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
        "Programming Language :: Python",
        "Topic :: Scientific/Engineering :: Biology",
    ],
)
