import os
import shutil
import numpy as np
import pretty_midi
from multiprocessing import Queue, Process
from pypianoroll import Multitrack
import argparse
import sys
import logging
logging.basicConfig(level=logging.INFO, stream=sys.stdout, format='%(asctime)s %(levelname)s: %(message)s')


def get_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--mid_root', type=str, default='/data00/home/zhangyonghui.98k/Dataset/lpd_5/lpd5_all')
    parser.add_argument('--dst_path', type=str, default='/data00/home/zhangyonghui.98k/Dataset/lpd5.npz')
    parser.add_argument('--num_workers', type=int, default=25)
    parser.add_argument('--sample_num', type=int, default=12)
    args = parser.parse_args()
    return args


def mid_to_npz(mid_file, sample_num=None, n_tracks=5, beat_resolution=12):

    default_inst = {
        'Drums': pretty_midi.Instrument(program=0, is_drum=True, name="Drums"),
        'Piano': pretty_midi.Instrument(program=0, is_drum=False, name="Piano"),
        'Guitar': pretty_midi.Instrument(program=24, is_drum=False, name="Guitar"),
        'Bass': pretty_midi.Instrument(program=32, is_drum=False, name="Bass"),
        'Strings': pretty_midi.Instrument(program=48, is_drum=False, name="Strings")
    }

    # add note to midi.
    mid = pretty_midi.PrettyMIDI(mid_file)
    valid_inst_num = 0
    for i, inst in enumerate(mid.instruments):
        if len(inst.notes) != 0:
            default_inst[inst.name] = True
            valid_inst_num += 1

    if valid_inst_num < n_tracks:
        return None

    for inst_name in default_inst:
        if default_inst[inst_name] is not True:
            inst = default_inst[inst_name]
            inst.notes.append(pretty_midi.Note(pitch=0, start=0, end=1, velocity=100))
            mid.instruments.append(inst)

    default_inst = ['Drums', 'Piano', 'Guitar', 'Bass', 'Strings']
    inst_idx = [-1] * len(default_inst)
    for i, inst_name in enumerate(default_inst):
        for k, d in enumerate(mid.instruments):
            if d.name == inst_name:
                inst_idx[i] = k

    assert np.sum(inst_idx) == 10, 'Some Instrument not existed!'

    label_mat = np.zeros((int((len(mid.get_beats())) * beat_resolution + 1), 128, 5), dtype=np.bool)
    time_ratio = label_mat.shape[0] / mid.get_end_time()
    if label_mat.shape[0] <= 8 * 4 * beat_resolution:
        logging.info('skip too short audio.')
        return None

    for write_id, inst_id in enumerate(inst_idx):
        inst = mid.instruments[inst_id]
        for note in inst.notes:
            start = int(note.start * time_ratio)
            end = int(note.end * time_ratio) + 1
            if end <= label_mat.shape[0]:
                label_mat[start:end, note.pitch, write_id] = True

    label_mat = label_mat[:, 24:108]

    # print('now.shape=', label_mat.shape)
    # print('now.sum=', np.sum(label_mat))
    result = []
    sample_start = [int(x * time_ratio) for x in mid.get_beats()]
    np.random.shuffle(sample_start)
    sample_start = sample_start[:sample_num]
    for start in sample_start:
        end = start + 8 * 4 * beat_resolution
        if end <= label_mat.shape[0]:
            sample = label_mat[start:end, ...]
            sample = np.reshape(sample, (8, 4 * beat_resolution, 84, 5))
            result.append(sample)
    return result


def mid_to_npz_select(mid_file, sample_num=15, select_inst=('Drums', 'Piano'), beat_resolution=12):

    default_inst = ['Drums', 'Piano', 'Guitar', 'Bass', 'Strings']
    write_idx = [-1] * len(select_inst)
    for i, inst_name in enumerate(select_inst):
        for j, d in enumerate(default_inst):
            if inst_name == d:
                write_idx[i] = j
                break

    mid = pretty_midi.PrettyMIDI(mid_file)
    select_idx = [-1] * len(select_inst)
    for k, inst in enumerate(mid.instruments):
        for i in range(len(select_inst)):
            if select_inst[i] == inst.name and len(inst.notes) != 0:
                select_idx[i] = k
                break

    for i, idx in enumerate(select_idx):
        if idx == -1:
            # logging.info('does not contain the instrument:{}'.format(select_inst[i]))
            return None
    label_mat = np.zeros((int((len(mid.get_beats())) * beat_resolution + 1), 128, 5), dtype=np.bool)
    time_ratio = label_mat.shape[0] / mid.get_end_time()
    if label_mat.shape[0] <= 8 * 4 * beat_resolution:
        logging.info('skip too short audio.')
        return None
    for inst_id, write_id in zip(select_idx, write_idx):
        inst = mid.instruments[inst_id]
        for note in inst.notes:
            start = int(note.start * time_ratio)
            end = int(note.end * time_ratio) + 1
            if end <= label_mat.shape[0]:
                label_mat[start:end, note.pitch, write_id] = True

    label_mat = label_mat[:, 24:108]

    # print('now.shape=', label_mat.shape)
    # print('now.sum=', np.sum(label_mat))
    result = []
    sample_start = [int(x * time_ratio) for x in mid.get_beats()]
    np.random.shuffle(sample_start)
    sample_start = sample_start[:sample_num]
    for start in sample_start:
        end = start + 8 * 4 * beat_resolution
        if end <= label_mat.shape[0]:
            sample = label_mat[start:end, ...]
            sample = np.reshape(sample, (8, 4 * beat_resolution, 84, 5))
            result.append(sample)
    return result


def check_mid(mid_path, gamma=5):
    mid = pretty_midi.PrettyMIDI(mid_path)
    # check inst num
    inst_num = 0
    for i, inst in enumerate(mid.instruments):
        if len(inst.notes) == 0:
            mid.instruments[i].notes.append(pretty_midi.Note(pitch=23, start=1, end=2, velocity=100))
    # if inst_num < gamma:
    #     logging.info('too less instruments.')
    #     return None
    p_idx = [int(x * 20) for x in mid.get_beats()]
    return p_idx


def send_mid(args, q):

    mid_files = [os.path.join(args.mid_root, x) for x in os.listdir(args.mid_root)]
    for i, mid_path in enumerate(mid_files):
        q.put(mid_path)
        if (i + 1) % 100 == 0:
            logging.info('processing: [{}/{}]'.format(i + 1, len(mid_files)))

    for _ in range(args.num_workers):
        q.put(None)


def processer(q_in, q_out, sample_num=None):

    while True:
        mid_path = q_in.get()
        if mid_path is None:
            q_out.put(None)
            break

        res = mid_to_npz(mid_path, sample_num)
        # res = mid_to_npz_select(mid_path, sample_num)
        if res is None:
            continue
        q_out.put(res)


def get_ary(args, q):

    get_none = 0
    result = []
    while True:
        res = q.get()
        if res is None:
            get_none += 1
            if get_none == args.num_workers:
                break
            continue
        result.extend(res)

    saved = np.asarray(result)
    logging.info('saved shape={}'.format(saved.shape))
    logging.info('save file into: {}'.format(args.dst_path))
    np.savez_compressed(args.dst_path, nonzero=np.array(saved.nonzero()), shape=saved.shape)


def work(args):

    if os.path.isfile(args.dst_path):
        logging.info('delete file: {}'.format(args.dst_path))
        os.remove(args.dst_path)

    q_in = Queue(maxsize=1024)
    q_out = Queue(maxsize=1024)

    sender = Process(target=send_mid, args=(args, q_in))
    sender.start()

    worker = [Process(target=processer, args=(q_in, q_out, args.sample_num)) for _ in range(args.num_workers)]
    for w in worker:
        w.start()

    geter = Process(target=get_ary, args=(args, q_out))
    geter.start()

    sender.join()
    for w in worker:
        w.join()
    geter.join()

    logging.info('finished.')


def main():
    args = get_args()
    logging.info(args)
    work(args)


if __name__ == '__main__':
    main()

