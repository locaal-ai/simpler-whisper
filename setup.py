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
        # This is the critical change - we need to get the proper extension suffix
        ext_suffix = sysconfig.get_config_var('EXT_SUFFIX')

        # Get the full path where the extension should be placed
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))

        # Ensure the extension directory exists
        os.makedirs(extdir, exist_ok=True)

        # Get acceleration and platform from environment variables
        acceleration = os.environ.get('SIMPLER_WHISPER_ACCELERATION', 'cpu')
        target_platform = os.environ.get('SIMPLER_WHISPER_PLATFORM', platform.machine())

        cmake_args = [
            f'-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}',
            f'-DPYTHON_EXTENSION_SUFFIX={ext_suffix}',  # Pass the extension suffix to CMake
            f'-DACCELERATION={acceleration}',
        ]

        env = os.environ.copy()

        # Add platform-specific arguments
        if platform.system() == "Darwin":  # macOS
            cmake_args += [
                f'-DCMAKE_OSX_ARCHITECTURES={target_platform}',
                '-DCMAKE_BUILD_WITH_INSTALL_RPATH=ON',
                '-DCMAKE_INSTALL_RPATH_USE_LINK_PATH=ON',
                f'-DCMAKE_INSTALL_NAME_DIR=@rpath'
            ]
            env["MACOS_ARCH"] = target_platform

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

        env['CXXFLAGS'] = f'{env.get("CXXFLAGS", "")} -DVERSION_INFO=\\"{self.distribution.get_version()}\\"'

        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)

        print("CMake args:", cmake_args)
        print("Build args:", build_args)
        print(f"Extension will be built in: {extdir}")

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
    packages=['simpler_whisper'],  # Add this line to ensure the package directory is created
)
