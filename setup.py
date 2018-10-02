import setuptools

setuptools.setup(
    name='keras_yolo3',
    version='0.0.1',
    author='unknown',
    author_email='unknown',
    description='Keras implementation of Yolo3.',
    maintainer='Sebastian Hartwig',
    maintainer_email='sebastian.hartwig@uni-ulm.de',
    packages=setuptools.find_packages(),
    install_requires=['keras', 'matplotlib']
)
