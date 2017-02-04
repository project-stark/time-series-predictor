from distutils.core import setup

setup(
    name='Home Assistant RTA',
    version='0.1',
    packages=['ts_predictor'],
    url='https://github.com/project-stark/time-series-predictor',
    license='',
    author='Divyesh Peshavaria',
    author_email='divyeshpeshavaria@gmail.com',
    description='',
    requires=[
        'numpy',
        'pandas',
        'scikit-learn'
    ]
)
