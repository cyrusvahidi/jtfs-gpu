from distutils.core import setup

setup(
    name="kymjtfs",
    version="0.1",
    description="",
    author="Cyrus Vahidi",
    author_email="c.vahidi@qmul.ac.uk",
    include_package_data=True,
    url="https://github.com/rastegah/kymatio-jtfs,",
    packages=["gin-config",
              "librosa", 
              "nnAudio", 
              "torch",
              "kymatio",
              "torchaudio",
              "tqdm",
              "mirdata",
              "pytorch-lightning"],
)