from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import sys
import os
import subprocess
import platform
import sysconfig
from wheel.bdist_wheel import bdist_wheel


class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=""):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


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
        # This is the critical change - we need to get the proper extension suffix
        ext_suffix = sysconfig.get_config_var("EXT_SUFFIX")

        # Get the full path where the extension should be placed
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))

        # Ensure the extension directory exists
        os.makedirs(extdir, exist_ok=True)

        # Get acceleration and platform from environment variables
        acceleration = os.environ.get("SIMPLER_WHISPER_ACCELERATION", "cpu")
        target_platform = os.environ.get("SIMPLER_WHISPER_PLATFORM", platform.machine())
        python_version = os.environ.get(
            "SIMPLER_WHISPER_PYTHON_VERSION",
            f"{sys.version_info.major}.{sys.version_info.minor}",
        )

        cmake_args = [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}",
            f"-DPYTHON_EXTENSION_SUFFIX={ext_suffix}",  # Pass the extension suffix to CMake
            f"-DACCELERATION={acceleration}",
            f"-DPYBIND11_PYTHON_VERSION={python_version}",
        ]

        env = os.environ.copy()

        # Add platform-specific arguments
        if platform.system() == "Darwin":  # macOS
            cmake_args += [
                f"-DCMAKE_OSX_ARCHITECTURES=arm64;x86_64",
                "-DCMAKE_BUILD_WITH_INSTALL_RPATH=ON",
                "-DCMAKE_INSTALL_RPATH_USE_LINK_PATH=ON",
                f"-DCMAKE_INSTALL_NAME_DIR=@rpath",
            ]
            # Remove the MACOS_ARCH environment variable as we're building universal
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

        print("CMake args:", cmake_args)
        print("Build args:", build_args)
        print(f"Extension will be built in: {extdir}")
        print(
            f"Building for Python {python_version} on {target_platform} with acceleration: {acceleration}"
        )

        subprocess.check_call(
            ["cmake", ext.sourcedir] + cmake_args, cwd=self.build_temp, env=env
        )
        subprocess.check_call(
            ["cmake", "--build", "."] + build_args, cwd=self.build_temp
        )


acceleration = os.environ.get("SIMPLER_WHISPER_ACCELERATION", "")


class CustomBdistWheel(bdist_wheel):
    def get_tag(self):
        python, abi, platform = super().get_tag()
        if acceleration:
            # Store original version
            orig_version = self.distribution.get_version()
            # Temporarily modify version
            self.distribution.metadata.version = f"{orig_version}+{acceleration}"
        return python, abi, platform


# Make version
pkg_version = "0.2.2"
if platform.system() == "Windows" and acceleration:
    pkg_version = f"{pkg_version}+{acceleration}"

setup(
    name="simpler-whisper",
    version=pkg_version,
    author="Roy Shilkrot",
    author_email="roy.shil@gmail.com",
    description="A simple Python wrapper for whisper.cpp",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    ext_modules=[CMakeExtension("simpler_whisper._whisper_cpp")],
    cmdclass={
        "build_ext": CMakeBuild,
        "bdist_wheel": CustomBdistWheel,
    },
    zip_safe=False,
    packages=[
        "simpler_whisper"
    ],  # Add this line to ensure the package directory is created
    python_requires=">=3.10",
    install_requires=[
        "numpy",
    ],
    package_data={
        "simpler_whisper": ["*.dll", "*.pyd", "*.so", "*.metal"],
    },
)
