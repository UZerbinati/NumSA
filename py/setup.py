from setuptools import setup
setup(
    name='numsa',
    version='0.0.1',    
    description='Numerical ToolBOX',
    url='',
    author='Umberto Zerbinati',
    author_email='umberto.zerbinati@kaust.edu.sa',
    license='LPGL',
    packages=['numsa'],
    install_requires=['numpy',
    		  'pygmsh',
    		  'pytest',
    		  'tensorflow'],
)
