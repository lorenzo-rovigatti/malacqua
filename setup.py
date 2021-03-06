from setuptools import setup, find_packages

PACKAGE_NAME = "malacqua"

setup(
    name=PACKAGE_NAME,
    use_scm_version={
        "fallback_version": "0.0.1",
        },
    packages=find_packages(),
    setup_requires=[
        'setuptools_scm',
        'numpy',
        'sklearn',
        'pandas',
        'matplotlib',
        'tensorflow'
        ],
    author='Lorenzo Rovigatti',
    author_email='lorenzo.rovigatti@uniroma1.it',
    url='https://github.com/lorenzo-rovigatti/malacqua',
    description='To be added',
    long_description=open("./README.md", 'r').read(),
    long_description_content_type="text/markdown",
    license='GNU GPL 3.0',
    zip_safe=False,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Environment :: Console",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
        "Topic :: Scientific/Engineering :: Physics"
        ]
)
