import argparse
import os

import numpy as np
from pycocotools.coco import COCO

def run(coco, cat_ids, output_dir, num_examples):
    object_scales = {
        1: 0.25,
        2: 0.28,
        3: 0.22,
        4: 0.26,
    }
    
    cats = {cat['id']: cat for cat in coco.dataset['categories']}
    
    for cat_id in cat_ids:
        cat_name = cats[cat_id]['name']
        print('generating {} poses for {}'.format(num_examples, cat_name))

        object_scale = object_scales[cat_id]
        position_mean = [0, 0, object_scale * 3]
        position_cov = np.diag(np.square(object_scale * 0.5 * np.array([1.5, 0.8, 2])))

        np.random.seed(0)
        position = np.random.multivariate_normal(position_mean, position_cov, num_examples)
        orientation = np.random.randn(num_examples, 4)
        orientation = np.divide(orientation, np.linalg.norm(orientation, axis=1, keepdims=True))
        poses = np.append(position, orientation, axis=1)
        
        output_path = os.path.join(output_dir, '{}.txt'.format(cat_name))
        print('saving poses to {}'.format(output_path))
        np.savetxt(output_path, poses)
        print('')


parser = argparse.ArgumentParser()
parser.add_argument('--coke_data_root', default='../../data/CokeDataset/', help='location of Coke dataset')
parser.add_argument('--ann_file', default='20190517_Coke.json')
parser.add_argument('--cat_ids', default='1,2,3,4', help='comma-separated category ids')
parser.add_argument('--num_examples', type=int, default=1000000, help='number of examples per category')
parser.add_argument('--output_dir', default='../data/cache/pose_lists/')
args = parser.parse_args()

if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)

coco = COCO(os.path.join(args.coke_data_root, 'annotations', args.ann_file))
cat_ids = [int(c.strip()) for c in args.cat_ids.split(',')]
run(coco, cat_ids, args.output_dir, args.num_examples)
