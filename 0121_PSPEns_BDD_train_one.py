# run_experiment.py
import click

# from src.pytorch_selection import *
# pytorch_init()

from src.paths import *
from src.pipeline import *
from src.a01_sem_seg import *

# 是否使用线程加载数据
b_threaded = False

# 使用的实验类（可以替换为你自己的类）
exp_class = ExpSemSegPSP_Ensemble_BDD

@click.command()
@click.argument('ensemble_id', type=int)
def main(ensemble_id):
    # 添加实验配置（可设置 experiment 名字）
    cfg = add_experiment(exp_class.cfg,
                         name="{norig}_{i:02d}".format(norig=exp_class.cfg['name'], i=ensemble_id),
                         )
    exp = exp_class(cfg)
    print_cfg(cfg)

    # 初始化数据集
    exp.init_default_datasets(b_threaded)

    # 如果只想快速测试训练流程，可以启用这两句限制数据集长度
    # exp.datasets['val'].set_fake_length(20)
    # exp.datasets['train'].set_fake_length(20)

    # 初始化训练流程
    exp.init_net("train")
    exp.init_transforms()
    exp.init_loss()
    exp.init_log()
    exp.init_pipelines()

    if b_threaded:
        exp.pipelines['train'].loader_class = SamplerThreaded
        exp.pipelines['val'].loader_class = SamplerThreaded

    # 启动训练
    exp.training_run()

if __name__ == '__main__':
    import sys
    sys.argv = ['0121_PSPEns_BDD_train_one.py', '1']
    main()

