from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import sys
import os
import subprocess
import platform
import sysconfig

class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=''):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)

class CMakeBuild(build_ext):
    def run(self):
        try:
            out = subprocess.check_output(['cmake', '--version'])
        except OSError:
            raise RuntimeError("CMake must be installed to build the following extensions: " +
                               ", ".join(e.name for e in self.extensions))

        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        
        # Correctly identify the Python executable and other Python-related paths
        if platform.system() == "Windows":
            python_executable = os.path.join(sys.prefix, 'python.exe')
        else:
            python_executable = os.path.join(sys.prefix, 'bin', 'python')
        
        python_include = sysconfig.get_path('include')
        python_lib = sysconfig.get_config_var('LIBDIR')
        
        import numpy
        numpy_include = numpy.get_include()
        
        cmake_args = [
            f'-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}',
            f'-DPYTHON_EXECUTABLE={python_executable}',
            f'-DPYTHON_INCLUDE_DIR={python_include}',
            f'-DNUMPY_INCLUDE_DIR={numpy_include}',
            '-DACCELERATION=cpu',
        ]

        cfg = 'Debug' if self.debug else 'Release'
        build_args = ['--config', cfg]

        if platform.system() == "Windows":
            cmake_args += [f'-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{cfg.upper()}={extdir}']
            if sys.maxsize > 2**32:
                cmake_args += ['-A', 'x64']
            build_args += ['--', '/m']
        else:
            cmake_args += [f'-DCMAKE_BUILD_TYPE={cfg}']
            build_args += ['--', '-j2']

        env = os.environ.copy()
        env['CXXFLAGS'] = f'{env.get("CXXFLAGS", "")} -DVERSION_INFO=\\"{self.distribution.get_version()}\\"'
        
        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)
        
        print("CMake args:", cmake_args)
        print("Build args:", build_args)
        print("Python executable:", python_executable)
        print("Python include:", python_include)
        print("Python lib:", python_lib)
        print("NumPy include:", numpy_include)
        
        subprocess.check_call(['cmake', ext.sourcedir] + cmake_args, cwd=self.build_temp, env=env)
        subprocess.check_call(['cmake', '--build', '.'] + build_args, cwd=self.build_temp)

setup(
    name='simpler-whisper',
    version='0.1.0',
    author='Your Name',
    author_email='your.email@example.com',
    description='A simple Python wrapper for whisper.cpp',
    long_description='',
    ext_modules=[CMakeExtension('simpler_whisper._whisper_cpp')],
    cmdclass=dict(build_ext=CMakeBuild),
    zip_safe=False,
)