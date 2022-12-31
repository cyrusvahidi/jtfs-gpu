from setuptools import setup

setup(
    name="kymjtfs",
    version="0.1",
    description="",
    author="Cyrus Vahidi",
    author_email="c.vahidi@qmul.ac.uk",
    include_package_data=True,
    packages=['kymjtfs'],
    url="https://github.com/cyrusvahidi/jtfs-gpu",
    install_requires=["gin-config",
              "librosa", 
              "nnAudio", 
              "torch",
              "kymatio",
              "torchaudio",
              "tqdm",
              "mirdata",
              "pytorch-lightning"],
)
