from setuptools import find_packages
import os

from distutils.core import setup

#ver_file = os.path.join("farmgym", "_version.py")
#with open(ver_file) as f:
#    exec(f.read())

packages = find_packages()

setup(name='average-reward-reinforcement-learning',
      version='v1',
      packages=packages,
      install_requires=['gym', 'numpy','scipy', 'joblib', 'matplotlib', 'networkx']  # And any other dependencies foo needs
)
#
# setup(
#     name="rlberry",
#     version=__version__,
#     description="An easy-to-use reinforcement learning library for research and education",
#     long_description=long_description,
#     long_description_content_type="text/markdown",
#     author="Omar Darwiche Domingues, Yannis Flet-Berliac, Edouard Leurent, Pierre Menard, Xuedong Shang",
#     url="https://github.com/rlberry-py",
#     license="MIT",
#     packages=packages,
#     classifiers=[
#         "Development Status :: 4 - Beta",
#         "Intended Audience :: Science/Research",
#         "Intended Audience :: Education",
#         "Topic :: Scientific/Engineering :: Artificial Intelligence",
#         "Programming Language :: Python :: 3.8",
#         "Programming Language :: Python :: 3 :: Only",
#         "License :: OSI Approved :: MIT License",
#         "Operating System :: OS Independent",
#     ],
#     install_requires=install_requires,
#     extras_require=extras_require,
#     zip_safe=False,
#)



# run with: pip install -e .