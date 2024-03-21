from setuptools import setup, find_packages
import sys
 
# Metadata
PROJECT_NAME = "tim_utils"
AUTHOR = "mgnigh98"
AUTHOR_EMAIL = "mgnigh98@gmail.com"
GITHUB_URL = f"https://github.com/{AUTHOR}/{PROJECT_NAME}"
DESCRIPTION = "Package for common imports and utility functions"
REQUIRED_PACKAGES = [line for line in open("requirements.txt").readlines() if 'https://' not in line]
REQUIRED_LINKS = [line.strip().split()[-1] for line in open("requirements.txt").readlines() if 'https://' in line]
VERSION = "0.1.0"
 
 
setup(
  name=PROJECT_NAME,
  version=VERSION,
  author=AUTHOR,
  author_email=AUTHOR_EMAIL,
  url=GITHUB_URL,
  description=DESCRIPTION,
  packages=find_packages(),
  install_requires=REQUIRED_PACKAGES,
  #dependency_links=REQUIRED_LINKS,
  classifiers=[
    # Choose your license from the "License" classifiers list:
    # https://pypi.org/classifiers/
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3",
    "License :: Free For Educational Use",
    "Topic :: Scientific/Engineering",
    "Development Status :: 4 - Beta",
  ],
  python_requires=f">={sys.version.split()[0]}",
)