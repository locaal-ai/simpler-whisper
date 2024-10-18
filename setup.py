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
        
        # Get acceleration and platform from environment variables
        acceleration = os.environ.get('SIMPLER_WHISPER_ACCELERATION', 'cpu')
        target_platform = os.environ.get('SIMPLER_WHISPER_PLATFORM', platform.machine())

        if os.environ.get('Python_ROOT_DIR') is not None:
            python_root_dir = os.environ.get('Python_ROOT_DIR')
            if platform.system() == "Windows":
                python_executable = os.path.join(python_root_dir, 'python.exe')
                python_include = os.path.join(python_root_dir, 'include')
                python_lib = os.path.join(python_root_dir, 'libs')
            else:
                python_executable = os.path.join(python_root_dir, 'bin', 'python')
                python_include = os.path.join(python_root_dir, 'include')
                python_lib = os.path.join(python_root_dir, 'lib')
        else:
            # Correctly identify the Python executable and other Python-related paths
            if platform.system() == "Windows":
                python_executable = sys.executable
            else:
                python_executable = os.path.join(sys.exec_prefix, 'bin', 'python')
            
            python_include = sysconfig.get_path('include')
            python_lib = sysconfig.get_config_var('LIBDIR')
        
        import numpy
        numpy_include = numpy.get_include()
        
        cmake_args = [
            f'-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}',
            f'-DPython_EXECUTABLE={python_executable}',
            f'-DPython_INCLUDE_DIR={python_include}',
            f'-DPython_LIBRARY={python_lib}',
            f'-DNUMPY_INCLUDE_DIR={numpy_include}',
            f'-DACCELERATION={acceleration}',
        ]

        # Add platform-specific arguments
        if platform.system() == "Darwin":  # macOS
            cmake_args.append(f'-DCMAKE_OSX_ARCHITECTURES={target_platform}')
            # add MACOS_ARCH env variable to specify the target platform
            os.environ["MACOS_ARCH"] = target_platform

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
    author='Roy Shilkrot',
    author_email='roy.shil@gmail.com',
    description='A simple Python wrapper for whisper.cpp',
    long_description='',
    ext_modules=[CMakeExtension('simpler_whisper._whisper_cpp')],
    cmdclass=dict(build_ext=CMakeBuild),
    zip_safe=False,
)