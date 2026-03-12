import re
import os
import cv2
import glob
import argparse
import numpy as np
from tqdm import tqdm
from functools import partial
from multiprocessing import Pool


def parse_split_file(anno_path):
    """
    Parse a SI or US annotation file.

    SI format (2 columns):  id|gloss
    US format (3 columns):  id|gloss|text

    Returns a list of (video_id, gloss) tuples in file order.
    """
    with open(anno_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    entries = []
    for line in lines[1:]:   # skip header row
        line = line.strip()
        if not line:
            continue
        parts = line.split('|')
        video_id = parts[0]
        gloss = parts[1]
        entries.append((video_id, gloss))
    return entries


def build_info_dict(entries, frames_root):
    """
    Build the info dict expected by BaseFeeder.

    The 'folder' value is a glob pattern relative to dataset_root (frames_root).
    For video id '00_0001' the frames live at:
        {frames_root}/00/00_0001/frame*.jpg
    so folder = '00/00_0001/frame*.jpg'.
    """
    info_dict = {}
    for file_idx, (video_id, gloss) in enumerate(entries):
        speaker = video_id.split('_')[0]
        folder = f"{speaker}/{video_id}/frame*.jpg"
        full_pattern = os.path.join(frames_root, folder)
        num_frames = len(glob.glob(full_pattern))
        info_dict[file_idx] = {
            'fileid': video_id,
            'folder': folder,
            'signer': speaker,
            'label': gloss,
            'num_frames': num_frames,
            'original_info': f"{video_id}|{gloss}",
        }
    return info_dict


def generate_gt_stm(info, save_path):
    with open(save_path, 'w', encoding='utf-8') as f:
        for k, v in info.items():
            if not isinstance(k, int):
                continue
            f.write(f"{v['fileid']} 1 {v['signer']} 0.0 1.79769e+308 {v['label']}\n")


def sign_dict_update(total_dict, info):
    for k, v in info.items():
        if not isinstance(k, int):
            continue
        for gloss in v['label'].split():
            total_dict[gloss] = total_dict.get(gloss, 0) + 1
    return total_dict


def resize_img(img_path, dsize):
    dsize_tuple = tuple(int(x) for x in re.findall(r'\d+', dsize))
    img = cv2.imread(img_path)
    if img is None:
        print(f'Warning: could not read {img_path}')
        return None
    return cv2.resize(img, dsize_tuple, interpolation=cv2.INTER_LANCZOS4)


def resize_video(video_idx, dsize, info_dict, frames_root, target_root):
    info = info_dict[video_idx]
    src_pattern = os.path.join(frames_root, info['folder'])
    img_list = sorted(glob.glob(src_pattern))
    speaker = info['signer']
    vid = info['fileid']
    dst_dir = os.path.join(target_root, speaker, vid)
    # Skip if already done
    if len(glob.glob(os.path.join(dst_dir, 'frame*.jpg'))) == len(img_list):
        return
    os.makedirs(dst_dir, exist_ok=True)
    for img_path in img_list:
        img = resize_img(img_path, dsize)
        if img is None:
            continue
        frame_name = os.path.basename(img_path)
        cv2.imwrite(os.path.join(dst_dir, frame_name), img)


def run_mp_cmd(processes, func, args):
    with Pool(processes) as p:
        list(tqdm(p.imap(func, args), total=len(args)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Preprocess the Isharah dataset for SlowFastSign training.')
    parser.add_argument('--dataset', type=str, default='Isharah',
                        help='output folder name under preprocess/')
    parser.add_argument('--frames-root', type=str, required=True,
                        help='Root directory containing speaker subdirs '
                             '(e.g. .../isharah1000/Volumes/SarahAlyami/isharah500)')
    parser.add_argument('--anno-root', type=str, required=True,
                        help='Root directory containing SI/ and US/ annotation folders '
                             '(e.g. .../isharah1000)')
    parser.add_argument('--split-type', type=str, default='SI', choices=['SI', 'US'],
                        help='SI: gloss-only annotations; US: gloss + Arabic text')
    parser.add_argument('--output-res', type=str, default='256x256px',
                        help='Target resolution when --process-image is set')
    parser.add_argument('--process-image', '-p', action='store_true', default=False,
                        help='Resize frames to --output-res and save to --target-root')
    parser.add_argument('--target-root', type=str, default=None,
                        help='Destination root for resized frames '
                             '(required when --process-image is used)')
    parser.add_argument('--multiprocessing', '-m', action='store_true', default=False,
                        help='Use multiprocessing to speed up frame resizing')

    args = parser.parse_args()

    if args.process_image and args.target_root is None:
        parser.error('--target-root is required when --process-image is set')

    os.makedirs(f'./{args.dataset}', exist_ok=True)
    sign_dict = {}

    # ------------------------------------------------------------------ #
    # Process each split                                                   #
    # ------------------------------------------------------------------ #
    all_info = {}
    for md in ['train', 'dev', 'test']:
        anno_path = os.path.join(args.anno_root, args.split_type, f'{md}.txt')
        if not os.path.exists(anno_path):
            print(f'WARNING: annotation file not found: {anno_path}')
            continue
        entries = parse_split_file(anno_path)
        info = build_info_dict(entries, args.frames_root)
        all_info[md] = info

        np.save(f'./{args.dataset}/{md}_info.npy', info)
        generate_gt_stm(info, f'./{args.dataset}/{args.dataset}-groundtruth-{md}.stm')
        sign_dict_update(sign_dict, info)
        print(f'{md}: {len(info)} samples')

    # ------------------------------------------------------------------ #
    # Optional: resize all frames                                          #
    # ------------------------------------------------------------------ #
    if args.process_image:
        # Flatten all samples across splits into a single dict for indexing
        flat_info = {}
        idx = 0
        for md in ['train', 'dev', 'test']:
            for v in all_info.get(md, {}).values():
                flat_info[idx] = v
                idx += 1
        video_indices = list(range(len(flat_info)))
        resize_fn = partial(
            resize_video,
            dsize=args.output_res,
            info_dict=flat_info,
            frames_root=args.frames_root,
            target_root=args.target_root,
        )
        print(f'Resizing {len(video_indices)} videos to {args.output_res} ...')
        if args.multiprocessing:
            run_mp_cmd(8, resize_fn, video_indices)
        else:
            for i in tqdm(video_indices):
                resize_fn(i)

    # ------------------------------------------------------------------ #
    # Build and save gloss dictionary                                      #
    # ------------------------------------------------------------------ #
    sign_dict_sorted = sorted(sign_dict.items(), key=lambda d: d[0])
    save_dict = {key: [idx + 1, value] for idx, (key, value) in enumerate(sign_dict_sorted)}
    np.save(f'./{args.dataset}/gloss_dict.npy', save_dict)
    print(f'Vocabulary size: {len(save_dict)}')
    print(f'Done. Outputs written to ./{args.dataset}/')
