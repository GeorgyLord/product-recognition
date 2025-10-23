# !pip install roboflow

from roboflow import Roboflow
rf = Roboflow(api_key="y4L5YxH0nQx0HWeCCljM")
project = rf.workspace("home-tvxgy").project("product-detected-c5igg")
version = project.version(8)
dataset = version.download("yolov12")
