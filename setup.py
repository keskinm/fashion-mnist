from setuptools import setup, find_packages

setup(
    name='fashion-mnist',
    packages=find_packages(),
    zip_safe=True,
    install_requires=[
        'matplotlib>=3.1.1,<4',
        'torch>=1.2.0,<2',
        'torchvision>=0.4.0,<1',
    ],
)
