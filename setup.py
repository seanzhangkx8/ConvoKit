from setuptools import setup

setup(
    name = "socialkit",
    author = "Andrew Wang",
    author_email = "azw7@cornell.edu",
    url = "https://github.com/qema/socialkit",
    description = "Social features toolkit",
    version = "0.0.1",
    packages = ["socialkit"],
    package_data = {"socialkit": ["data/*"]},
    entry_points = {
        "console_scripts": ["socialkit = socialkit:command_line_main"]
    }
)
