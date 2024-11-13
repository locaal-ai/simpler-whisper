from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
from setuptools.command.build_py import build_py
import sys
import os
import subprocess
import platform
import sysconfig


class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=""):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class BuildPyCommand(build_py):
    def run(self):
        self.run_command("build_ext")
        return super().run()


class CMakeBuild(build_ext):
    def run(self):
        try:
            out = subprocess.check_output(["cmake", "--version"])
        except OSError:
            raise RuntimeError(
                "CMake must be installed to build the following extensions: "
                + ", ".join(e.name for e in self.extensions)
            )

        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        ext_suffix = sysconfig.get_config_var("EXT_SUFFIX")
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        os.makedirs(extdir, exist_ok=True)

        acceleration = os.environ.get("SIMPLER_WHISPER_ACCELERATION", "cpu")
        python_version = f"{sys.version_info.major}.{sys.version_info.minor}"

        cmake_args = [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}",
            f"-DPYTHON_EXTENSION_SUFFIX={ext_suffix}",
            f"-DACCELERATION={acceleration}",
            f"-DPYBIND11_PYTHON_VERSION={python_version}",
        ]

        env = os.environ.copy()

        if platform.system() == "Darwin":
            cmake_args += [
                "-DCMAKE_OSX_ARCHITECTURES=arm64;x86_64",
                "-DCMAKE_OSX_DEPLOYMENT_TARGET=10.13",
                "-DCMAKE_BUILD_WITH_INSTALL_RPATH=ON",
                "-DCMAKE_INSTALL_RPATH_USE_LINK_PATH=ON",
                f"-DCMAKE_INSTALL_NAME_DIR=@rpath",
            ]
            # Set environment variables for universal build
            env["MACOSX_DEPLOYMENT_TARGET"] = "10.13"
            env["_PYTHON_HOST_PLATFORM"] = "macosx-10.13-universal2"

            # Remove any existing arch flags that might interfere
            if "ARCHFLAGS" in env:
                del env["ARCHFLAGS"]
            if "MACOS_ARCH" in env:
                del env["MACOS_ARCH"]

        cfg = "Debug" if self.debug else "Release"
        build_args = ["--config", cfg]

        if platform.system() == "Windows":
            cmake_args += [f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{cfg.upper()}={extdir}"]
            if sys.maxsize > 2**32:
                cmake_args += ["-A", "x64"]
            build_args += ["--", "/m"]
        else:
            cmake_args += [f"-DCMAKE_BUILD_TYPE={cfg}"]
            build_args += ["--", "-j2"]

        env["CXXFLAGS"] = (
            f'{env.get("CXXFLAGS", "")} -DVERSION_INFO=\\"{self.distribution.get_version()}\\"'
        )

        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)

        subprocess.check_call(
            ["cmake", ext.sourcedir] + cmake_args, cwd=self.build_temp, env=env
        )
        subprocess.check_call(
            ["cmake", "--build", "."] + build_args, cwd=self.build_temp
        )


acceleration = os.environ.get("SIMPLER_WHISPER_ACCELERATION", "cpu")

setup(
    name="simpler-whisper",
    version=f"0.2.4+{acceleration}",
    author="Roy Shilkrot",
    author_email="roy.shil@gmail.com",
    description="A simple Python wrapper for whisper.cpp",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    ext_modules=[CMakeExtension("simpler_whisper._whisper_cpp")],
    cmdclass={
        "build_ext": CMakeBuild,
        "build_py": BuildPyCommand,
    },
    zip_safe=False,
    packages=["simpler_whisper"],
    python_requires=">=3.10",
    install_requires=[
        "numpy",
    ],
    package_data={
        "simpler_whisper": ["*.dll", "*.pyd", "*.so", "*.metal", "*.bin", "*.dylib"],
    },
)
