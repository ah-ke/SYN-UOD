from pathlib import Path
import os, json
from ..pipeline.frame import Frame
from .dataset import DatasetFrameList, ChannelLoaderImage
from ..paths import DIR_DSETS

DIR_ROAD_ANOMALY = Path(os.environ.get('DIR_ROAD_ANOMALY', DIR_DSETS / 'dataset_Fishyscapes'))


class DatasetFishyscapes(DatasetFrameList):
    name = 'fishyscapes'
    split = 'test'

    def __init__(self, dir_root=DIR_ROAD_ANOMALY, b_cache=True):
        img_list = sorted([p.name for p in (dir_root / 'images').glob('*.png')])

        self.img_ext = Path(img_list[0]).suffix
        self.dir_root = dir_root

        frames = [Frame(fid=Path(img_filename).stem) for img_filename in img_list]

        super().__init__(frames, b_cache=b_cache)

        self.add_channels(
            image=ChannelLoaderImage(
                img_ext=self.img_ext,
                file_path_tmpl='{dset.dir_root}/images/{fid}{channel.img_ext}',
            ),
            labels_source=ChannelLoaderImage(
                file_path_tmpl='{dset.dir_root}/labels_masks/{fid}.png',
            ),
        )

    @staticmethod
    def tr_get_anomaly_gt(labels_source, **_):
        return dict(
            anomaly_gt=labels_source >= 2,
        )

    @staticmethod
    def tr_get_roi(**_):
        return dict(roi=None)
