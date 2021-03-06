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
    package_data={'': ['*.so']},
    include_package_data=True,
    install_requires=['numpy',
    		  'tqdm',
    		  'matplotlib',
    		  'pytest',
    		  'mpi4py',
    		  'pytest-mpi'],
    extras_require={
    	'Hessian': ['tensorflow'],
    	'FEM': ['petsc4py',
    	        'slepc5py',
    	        'ngsolve'],
    }
)
