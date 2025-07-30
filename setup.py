from setuptools import setup, find_packages


# Function to read requirements from requirements.txt file
def parse_requirements(filename):
    with open(filename) as f:
        return f.read().splitlines()


# Read requirements from requirements.txt
requirements = parse_requirements("requirements.txt")

setup(
    name="msclip",
    version="0.1.0",
    description="Llama3-MS-CLIP",
    author="Clive Tinashe Marimo",
    packages=find_packages(),
    package_dir={"": "msclip"},
    python_requires=">=3.9",
    install_requires=requirements,  # Add requirements from requirements.txt
)
