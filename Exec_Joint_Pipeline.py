# main.py

# 导入依赖
from src.datasets.dataset import DatasetImageDir
from src.a05_differences.E1_article_evaluation import DiscrepancyJointPipeline
from src.a05_differences.E1_article_evaluation import DiscrepancyJointPipeline_LabelsOnly
from src.datasets.road_anomaly import DatasetRoadAnomaly

# 加载数据集
dset = DatasetImageDir(dir_root='data/joint_pipeline_example')
dset.discover()

# 初始化Joint Pipeline
joint_pipeline = DiscrepancyJointPipeline()
joint_pipeline.init_semseg()
joint_pipeline.init_gan()
joint_pipeline.init_discrepancy()
# 可选：启用半精度优化
# joint_pipeline.init_apex_optimization()
# 可选：调整batch size
joint_pipeline.set_batch_size(2)

# 运行并显示结果
joint_pipeline.run_on_dset(dset, b_show=True)

# 运行并保存结果
joint_pipeline.run_on_dset(dset, b_show=False)

# Alternative pipeline without image generator
joint_pipeline_lab = DiscrepancyJointPipeline_LabelsOnly()
joint_pipeline_lab.init_semseg()
joint_pipeline_lab.init_discrepancy()

# 加载另一个数据集
dset_road_anomaly = DatasetRoadAnomaly()
dset_road_anomaly.discover()
joint_pipeline_lab.run_on_dset(dset, b_show=True)
joint_pipeline_lab.run_on_dset(dset, b_show=False)
