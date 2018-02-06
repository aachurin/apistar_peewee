import sys
from setuptools import setup

setup(
    name='apistar-peewee',
    version='0.2.1',
    url='https://github.com/aachurin/apistar_peewee',
    license='BSD',
    author='Andrey Churin',
    author_email='aachurin@gmail.com',
    description='Peewee integration for Apistar',
    packages=['apistar_peewee'],
    zip_safe=False,
    platforms='any',
    install_requires=[
        'peewee >= 3.0.10',
        'apistar'
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Environment :: Web Environment',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: BSD License',
        'Operating System :: OS Independent',
        'Topic :: Internet :: WWW/HTTP :: Dynamic Content',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ]
)