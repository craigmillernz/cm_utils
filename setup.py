from setuptools import setup

setup(
   name='cm_utils',
   version='0.1',
   description='Various little functions',
   author='Craig Miller',
   author_email='c.miller@gns.cri.nz',
   packages=['cm_utils'],  #same as name
   install_requires=['matplotlib', 'numpy', 'scipy', 'shapefile'], #external packages as dependencies
)
