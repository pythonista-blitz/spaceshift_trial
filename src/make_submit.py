# -*- coding: utf-8 -*-
"""
requirements:
    python: 3.8.5
    library:
        numpy: 1.16.3
        opencv-python: 4.1.0
"""
import numpy as np
import os
import json
import argparse
import time
import cv2

def make_json(annotations_dir, categories):
    count = 0
    annotation_files = os.listdir(annotations_dir)
    json_data = {}
    s = time.time()
    for annotation_file in annotation_files:
        print(annotation_file)
        name = annotation_file.split('.')[0]
        json_data[name] = {}
        img = cv2.imread(os.path.join(annotations_dir, annotation_file), flags = cv2.IMREAD_UNCHANGED)
        for category in categories:
            category_segments = {}
            x, y = np.where(img==categories[category])
            category_pix = {}
            for i, j in zip(x, y):
                if i not in category_pix:
                    category_pix[i] = []
                category_pix[i].append(j)
            for l in category_pix:
                segments = []
                num_segments = 0
                for i,v in enumerate(sorted(category_pix[l])):
                    if i == 0:
                        start = v
                        end = v
                    else:
                        if v == end + 1:
                            end = v
                        else:
                            segments.append([int(start), int(end)])
                            start = v
                            end = v
                            num_segments += 1
                segments.append([int(start), int(end)])
                category_segments[int(l)]=segments
            if len(category_pix):
                json_data[name][category]=category_segments
        count+=1
        print(count, time.time()-s)

    return json_data

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--annotations_dir', type = str, help = 'directory of the annotations', nargs='?')
    args = parser.parse_args()

    categories = {'new_building':1}

    json_data = make_json(args.annotations_dir, categories)

    with open('submit.json', 'w') as f:
        json.dump(json_data, f, sort_keys=True,separators=(',', ':'))
