import argparse
from src.datasets.road_anomaly import DatasetRoadAnomaly
from src.datasets.fishyscapes import DatasetFishyscapes
from src.a05_differences.E1_article_evaluation import EvaluationDetectingUnexpected

# 异常图像 送入语义分割模型(19类)进行预测在利用预测的结果进行图像重构最后利用预训练的差异网络识别真实图像和重构图像之间的差异，从而获得异常分数
def main(args):
    # 加载数据集
    dset = DatasetRoadAnomaly()  # 可选参数：dir_root 指定数据集路径
    # dset = DatasetFishyscapes()
    dset.discover()

    # 初始化评估对象
    eval_obj = EvaluationDetectingUnexpected(sem_seg_variant='BaySegBdd')
    # eval_obj = EvaluationDetectingUnexpected(sem_seg_variant='PSPEnsBdd')

    # 运行语义分割
    eval_obj.init_semseg()
    eval_obj.run_semseg(dset, b_show=args.show_results)

    # 运行图像生成
    eval_obj.init_gen_image()
    eval_obj.run_gen_image(dset, b_show=args.show_results)

    # 运行异常检测
    eval_obj.run_detector('rbm', dset, b_show=args.show_results)
    eval_obj.run_detector('discrepancy_label_and_gen', dset, b_show=args.show_results)
    eval_obj.run_detector_all(dset)

    # # 生成示例图像demo
    # eval_obj.run_demo_imgs(dset, b_show=args.show_results)
    #
    # # 绘制ROC曲线
    # rocinfos = eval_obj.run_roc_curves_for_variant(dset)
    # # eval_obj.roc_plot_variants(dset)
    # eval_obj.roc_plot_variants(dset, rocinfos)

    print("Evaluation completed successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Road Anomaly Detection Evaluation")
    parser.add_argument("--show_results", action="store_true", help="Whether to display results")
    args = parser.parse_args()

    main(args)
