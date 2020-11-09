from setuptools import setup


setup(name='addon_seg_evaluation',
      version="0.1",
      packages=['addon_seg_evaluation'],
 	install_requires=['clickpoints', 'tensorflow', 'deformationcytometer', 'opencv-python']
      )
