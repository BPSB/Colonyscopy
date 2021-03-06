#!/usr/bin/python3

from setuptools import setup
from io import open

requirements = [
		'scipy',
		'numpy'
	]

setup(
		name = 'colonyscopy',
		description = 'Analysis of colony growth over time.',
		long_description = open('README.rst', encoding='utf8').read(),
		python_requires=">=3.3",
		packages = ['colonyscopy'],
		install_requires = requirements,
		setup_requires = ['setuptools_scm','pytest-runner'],
		tests_require = ['pytest'],
		use_scm_version = {'write_to': 'colonyscopy/version.py'},
		classifiers = [
				'Development Status :: Alpha',
				'License :: OSI Approved :: BSD License',
				'Operating System :: POSIX',
				'Operating System :: MacOS :: MacOS X',
				'Operating System :: Microsoft :: Windows',
				'Programming Language :: Python',
				'Topic :: Scientific/Engineering :: Biology',
			],
	)

