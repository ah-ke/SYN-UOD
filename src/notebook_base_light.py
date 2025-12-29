import numpy as np
import cv2 as cv
import torch
import matplotlib.pyplot as plt
import logging

from src.common.util_notebook import *
from src.pipeline.log import log
from src.pipeline.frame import Frame
from src.common.jupyter_show_image import show
from src.pipeline.transforms import TrByField, TrBase, TrsChain, TrKeepFields, TrAsType, TrKeepFieldsByPrefix, tr_print

# 设置 Matplotlib 默认风格
if not globals().get('PLT_STYLE_OVERRIDE'):
    plt.style.use('dark_background')
else:
    print('No plt style set')

plt.rcParams['figure.figsize'] = (12, 8)

# 兼容 Matplotlib 显示方式
try:
    import matplotlib
    matplotlib.use('Agg')  # 非交互模式，适用于 PyCharm 运行
except Exception as e:
    print(f"Warning: Unable to set Matplotlib backend: {e}")

# 定义一个显示图片的辅助函数，兼容 PyCharm
def display_image(img, title="Image", cmap=None):
    """ 在 PyCharm 兼容的环境下显示图片 """
    plt.figure()
    plt.imshow(img, cmap=cmap)
    plt.title(title)
    plt.axis("off")
    plt.show()

# 配置日志
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

log.info("Notebook base light script initialized successfully for PyCharm.")
