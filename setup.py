from setuptools import setup

setup(
    name='BlochHead',
    version='0.0.1',
    packages=['BlochHead'],
    url='https://gitlab.com/mtessmer/BlochHead',
    license='GNU GPLv3',
    author='Maxx Tessmer',
    author_email='mhtessmer@gmail.com',
    description='A Bloch simulator for EPR',
    install_requires=['numpy', 'scipy', 'matplotlib', 'PulseShape']
)
