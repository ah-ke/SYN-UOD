from src.a01_sem_seg import *

# 是否使用多线程的数据加载方式
b_threaded = False
dset = DatasetBDD_Segmentation(split='val', b_cache=b_threaded)
dset.discover()

# 初始化评估对象
exp = ExpSemSegBayes_BDD()

# 推理并评估
exp.init_net('eval')
exp.init_transforms()
exp.init_loss()
exp.init_log()
exp.init_pipelines()

# 如果启用了多线程加载，将训练和验证使用的 loader 替换为自定义的 SamplerThreaded
if b_threaded:
	exp.pipelines['val'].loader_class = SamplerThreaded

exp.pipelines['val'].execute(exp.datasets['val'], b_accumulate=False)
exp.datasets['val'].flush_hdf5_files()


