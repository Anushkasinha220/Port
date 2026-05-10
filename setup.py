from setuptools import setup, find_packages

setup(
    name="port_optimus",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "plotly",
        "streamlit",
    ],
)