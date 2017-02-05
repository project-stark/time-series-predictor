from distutils.core import setup

setup(
    name='ts_predictor',
    version='0.2',
    packages=['ts_predictor'],
    url='https://github.com/project-stark/time-series-predictor',
    license='',
    author='Divyesh Peshavaria',
    author_email='divyeshpeshavaria@gmail.com',
    description='',
    install_requires=[
        'numpy',
        'scipy',
        'pandas',
        'scikit-learn'
    ]
)
