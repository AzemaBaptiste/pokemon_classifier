from pkg_resources import parse_requirements
from setuptools import setup

REQUIREMENTS = [i.strip() for i in open("requirements.txt").readlines()]

setup(
    name='bazema_pokemon',
    author='Baptiste Azéma',
    author_email='baptiste@azema.tech',
    version='1.0',
    packages=['bazema_pokemon'],
    package_data={'bazema_pokemon.resources': ['*.pth']},
    include_package_data=True,
    python_requires='~=3.6',
    install_requires=REQUIREMENTS,
    description='Real-time application in order to dominate Humans and Pokémon.',
    license='LICENSE',
    entry_points={
        'console_scripts': ['bazema_pokemon=bazema_pokemon.__main__:main']
    },
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown'
)
