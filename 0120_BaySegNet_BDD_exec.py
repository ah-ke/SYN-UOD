#!/usr/bin/env python3

# from src.pytorch_selection import *
# pytorch_init()

from src.paths import *
from src.pipeline import *
from src.a01_sem_seg import *

# 是否使用多线程的数据加载方式
b_threaded = False

exp = ExpSemSegBayes_BDD()

# BDD100K 链接: https://pan.baidu.com/s/1Md9ErVPSPI0oIPWbzPrKhg 提取码： yang
exp.init_default_datasets(b_threaded)

# 可选: 用于调试或快速跑通流程
# exp.datasets['val'].set_fake_length(10)
# exp.datasets['train'].set_fake_length(10)

# 启动一个用于语义分割任务的贝叶斯神经网络训练流程
exp.init_net("train")
exp.init_transforms()
exp.init_loss()
exp.init_log()
exp.init_pipelines()
# 如果启用了多线程加载，将训练和验证使用的 loader 替换为自定义的 SamplerThreaded
if b_threaded:
	exp.pipelines['train'].loader_class = SamplerThreaded
	exp.pipelines['val'].loader_class = SamplerThreaded

# 模型训练
exp.training_run()


#qsub -jc 24h.1gpu /home/lis/programs/uge_run.sh "python /home/lis/dev/unknown_dangers/0109_EpistemicPSP_exec.py"


