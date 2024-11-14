import shutil
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

        cmake_args = [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}",
            f"-DPYTHON_EXTENSION_SUFFIX={ext_suffix}",
            f"-DACCELERATION={acceleration}",
            f"-DPYTHON_EXECUTABLE={sys.executable}",
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
        else:
            # Remove the existing CMakeCache.txt to ensure a clean build
            cache_file = os.path.join(self.build_temp, "CMakeCache.txt")
            if os.path.exists(cache_file):
                os.remove(cache_file)

        # Configure and build the extension
        subprocess.check_call(
            ["cmake", ext.sourcedir] + cmake_args, cwd=self.build_temp, env=env
        )
        subprocess.check_call(
            ["cmake", "--build", "."] + build_args, cwd=self.build_temp
        )


def get_latest_git_tag():
    tag = os.environ.get("SIMPLER_WHISPER_VERSION")
    if not tag:
        try:
            tag = subprocess.check_output(
                ["git", "describe", "--tags"], encoding="utf-8"
            ).strip()
        except subprocess.CalledProcessError:
            return "0.0.0-dev"
    parts = tag.split("-")
    if len(parts) == 3:
        return f"{parts[0]}-dev{parts[1]}"
    return tag


setup(
    name="simpler-whisper",
    version=get_latest_git_tag(),
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
        "simpler_whisper": [
            "./*.dll",
            "./*.pyd",
            "./*.so",
            "./*.metal",
            "./*.bin",
            "./*.dylib",
        ],
    },
    include_package_data=True,
)
