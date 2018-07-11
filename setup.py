from setuptools import setup, find_packages

setup(
    name='mushroombot-cloud-ml',
    version='1.0.0',
    description='Mushroom segmentation TensorFlow Application',
    include_package_data=True,
    packages=find_packages(exclude=['contrib', 'docs', 'tests']),
    install_requires=['tensorflow', 'lxml', 'pillow'],
)
