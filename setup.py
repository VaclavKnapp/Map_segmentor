from setuptools import setup, find_packages

setup(
    name="river-segmentation",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=1.9.0",
        "torchvision>=0.10.0",
        "transformers>=4.15.0",
        "albumentations>=1.1.0",
        "opencv-python>=4.5.4",
        "numpy>=1.20.0",
        "matplotlib>=3.4.3",
        "tqdm>=4.62.3",
        "Pillow>=8.4.0",
        "scikit-learn>=1.0.1",
    ],
    python_requires='>=3.7',
    author="Václav Knapp",
    author_email="vasikknapp@gmail.com",
    description="River segmentation from historical maps using SegFormer with Chamfer-IoU Loss",
    keywords="deep-learning, semantic-segmentation, river-segmentation, historical-maps",
    url="https://github.com/VaclavKnapp/Map_segmentor",
)
