from setuptools import setup

setup(
    name = "convokit",
    author = "Cristian Danescu-Niculescu-Mizil, Andrew Wang",
    author_email = "azw7@cornell.edu",
    url = "https://github.com/CornellNLP/Cornell-Conversational-Analysis-Toolkit",
    description = "Cornell Conversational Analysis Toolkit",
    version = "0.0.3",
    packages = ["convokit"],
    package_data = {"convokit": ["data/*.txt"]},
    entry_points = {
        "console_scripts": ["convokit = convokit.command_line:command_line_main"]
    },
    install_requires = [
        "matplotlib>=1.5.0",
        "pandas>=0.20.3",
        "spacy>=2.0.11",
        "scipy>=0.16.0",
        "scikit-learn>=0.16.1",
        "nltk>=3.3"
    ]
)
