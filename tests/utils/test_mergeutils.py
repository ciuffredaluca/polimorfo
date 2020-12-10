from pathlib import Path

import numpy as np
from pytest import fixture

from polimorfo.datasets import CocoDataset
from polimorfo.utils.mergeutils import get_img_meta_by_name, merge_datasets

BASE_PATH = Path(__file__).parent.parent / 'data'

@fixture
def dataset_file():
    return BASE_PATH / 'hair_drier_toaster_bear.json'

def test_merge_datasets():

    datasets = [ CocoDataset(coco_path=BASE_PATH / dataset_name) for dataset_name 
                    in ['dataset1.json', 'dataset2.json'] ]

    merged_ds = merge_datasets(datasets, BASE_PATH / 'fake_merge.json') 

    assert len(merged_ds.anns) == 20

    for k in ['scratch', 'dent']:
        assert merged_ds.count_images_per_category()[k] == np.sum([ds_item.count_images_per_category()[k] for ds_item in datasets])
        assert merged_ds.count_annotations_per_category()[k] == np.sum([ds_item.count_annotations_per_category()[k] for ds_item in datasets])

    # check image meta/ anns consistency
    for ds_item in datasets:
        for img_idx, img_meta in ds_item.imgs.items():

            anns = ds_item.get_annotations(img_idx)

            merged_img_meta = get_img_meta_by_name(merged_ds, img_meta['file_name'])
            merged_img_idx = merged_img_meta['id']
            merged_anns = merged_ds.get_annotations(merged_img_idx)
            # number of ann per image
            assert len(merged_anns) == len(anns)
            # number of cats per image
            merged_cats = { a['category_id'] for a in merged_anns }
            cats = { a['category_id'] for a in anns }
            assert len(merged_cats) == len(cats)
            # number of annotated points per image
            merged_segm_points = { len(a['segmentation']) for a in merged_anns }
            segm_points = { len(a['segmentation']) for a in anns }
            assert len(merged_segm_points) == len(segm_points)
            # annotation areas per image
            merged_areas = { a['area'] for a in merged_anns }
            areas = { a['area'] for a in anns }
            assert merged_areas == areas
        
def test_self_merge():
    f = BASE_PATH / 'dataset1.json'
    coco = CocoDataset(f)
    len_imgs_before = len(coco.imgs)
    len_anns_before = len(coco.anns)
    len_cats_before = len(coco.cats)
    print(len_anns_before)
    coco = merge_datasets([coco for _ in range(3)], BASE_PATH / 'fake_merge.json')
    assert len(coco.imgs) == len_imgs_before #1 loop returns 10 imgs, 2 loop returns 20, 3 loop returns 40
    assert len(coco.anns) == 3 * len_anns_before
    assert len(coco.cats) == len_cats_before

def test_merge_2_datasets():
    f1 = BASE_PATH / 'dataset1.json'
    f2 = BASE_PATH / 'dataset2.json'
    coco = CocoDataset(f1)
    coco2 = CocoDataset(f2)
    len_imgs_before = len(coco.imgs)
    len_anns_before = len(coco.anns)
    len_cats_before = len(coco.cats)
    coco = merge_datasets([coco, coco2], BASE_PATH / 'fake_merge.json')
    assert len(coco.imgs) == len_imgs_before + len(coco2.imgs)
    assert len(coco.anns) == len_anns_before + len(coco2.anns)
    assert len(coco.cats) == len_cats_before

def test_merge_heterogenius_datasets(dataset_file):
    f1 = BASE_PATH / 'dataset1.json'
    coco = CocoDataset(f1)
    coco2 = CocoDataset(dataset_file)
    len_imgs_before = len(coco.imgs)
    len_anns_before = len(coco.anns)
    len_cats_before = len(coco.cats)
    coco = merge_datasets([coco, coco2], BASE_PATH / 'fake_merge.json')
    assert len(coco.imgs) == len_imgs_before + len(coco2.imgs)
    assert len(coco.anns) == len_anns_before + len(coco2.anns)
    assert len(coco.cats) == len_cats_before + len(coco2.cats)
