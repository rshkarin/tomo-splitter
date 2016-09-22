import os
import sys
import re
import shutil
import argparse

import numpy as np
from skimage import io, filters, measure, feature
from sklearn.cluster import MeanShift, estimate_bandwidth

CAMERA_ANDOR = 'Andor'
CAMERA_PCO_DIMAX = 'dimax'

def read_andor_data_batch(files, shape=(2160,2568), roi_shape=(2160,2560), dtype=np.uint16):
    output = np.zeros((len(files),) + roi_shape, dtype=np.float32)

    for i,path in enumerate(files):
        output[i] = np.memmap(path, \
                              dtype=np.uint16, \
                              shape=shape, mode='r')[:roi_shape[0],:roi_shape[1]]

    return output

def _get_idx(name):
    v = re.findall('\d+', name)
    return int(v[0]) if len(v) else 0

def _get_name_idx(name):
    return _get_idx(name[::-1])

def camera_is_available(camera_type):
    return camera_type in [CAMERA_ANDOR, CAMERA_PCO_DIMAX]

def get_data_pathes(input_dir, camera_type, sep='@', batch_size=100):
    if not camera_is_available(camera_type):
        raise ValueError('Wrong camera type!')

    def ext_by_camera(name):
        return '.dat' if name == CAMERA_ANDOR else '.tif'

    files = [f for f in os.listdir(input_dir) if f.endswith(ext_by_camera(camera_type))]

    if camera_type == CAMERA_PCO_DIMAX:
        files.sort(key=lambda e: (sep in e, _get_idx(e)))
        files = [os.path.join(input_dir, f) for f in files]
    elif camera_type == CAMERA_ANDOR:
        files_andor = [(f, _get_name_idx(f)) for f in files]
        files_andor.sort(key=lambda e: e[1])
        files_andor = np.array([os.path.join(input_dir, f[0]) for f in files_andor])
        n_batches = int(np.ceil(len(files_andor) / float(batch_size)))
        files = np.array_split(files_andor, n_batches)
    else:
        raise ValueError('Wrong camera type!')

    return files

def estimate_profile(files, camera_type, patch_radius=16):
    profile = np.array([])
    height, width = None, None
    _roi = None

    if not camera_is_available(camera_type):
        raise ValueError('Wrong camera type!')

    for f in files:
        data = None

        if camera_type == CAMERA_PCO_DIMAX:
            data = io.imread(f)
        elif camera_type == CAMERA_ANDOR:
            data = read_andor_data_batch(f)
        else:
            raise ValueError('Wrong camera type!')

        if height is None or width is None:
            height, width = data.shape[1:]

        if _roi is None:
            hc, wc = int(height/2), int(width/2)
            _roi = np.index_exp[:,\
                                hc-patch_radius:hc+patch_radius+1,\
                                wc-patch_radius:wc+patch_radius+1]

        profile = np.append(profile, data[_roi].sum(axis=(1,2)))

    return profile

def split_profile(arr, shrinkage_ratio=50, frac_tolerance=0.1):
    idxs = np.arange(len(arr), dtype=np.int64)
    arr_shifted = np.append(arr[1:], arr[-1])
    diff = abs(arr - arr_shifted)

    def shrink_cond(arr, idx, shrinkage_ratio):
        prev_v = arr[idx-1] if idx > 0 else None
        next_v = arr[idx+1] if idx < len(arr)-1 else None

        if prev_v is None:
            return arr[idx] > next_v * shrinkage_ratio
        elif next_v is None:
            return arr[idx] > prev_v * shrinkage_ratio
        else:
             return arr[idx] > prev_v * shrinkage_ratio and \
                    arr[idx] > next_v * shrinkage_ratio

    def group_similar(arr, frac_tolerance):
        out = []

        for i in xrange(len(arr)):
            cur_grp = []

            cum_non_unique = []
            for j in xrange(i, len(arr)):
                non_unique = [j in grp for grp in out]
                cum_non_unique.extend(non_unique)

                if not np.any(non_unique) and \
                   abs(arr[i] - arr[j]) <= arr[i] * frac_tolerance:
                    cur_grp.append(j)

            if len(cur_grp):
                out.append(cur_grp)
                cum_non_unique = []

        return out

    def get_group_names(group_means):
        if len(group_means) != 3:
            raise ValueError("Something wrong, the dataset should have only 3 types of images.")

        sorted_means_idxs = np.argsort(group_means)
        names = np.array(['dark','proj','flat'])
        return zip(sorted_means_idxs, names)

    shrinked_arr = np.array([diff[i] if shrink_cond(diff, i, shrinkage_ratio) else 0 \
                    for i in xrange(len(diff))])
    ranges, ranges_idxs = [], []
    range_ends = np.flatnonzero(shrinked_arr)
    for i in xrange(len(range_ends) + 1):
        if i == 0:
            ranges.append(arr[:range_ends[i]+1])
            ranges_idxs.append(idxs[:range_ends[i]+1])
        elif i in xrange(len(range_ends)):
            ranges.append(arr[range_ends[i-1]+1:range_ends[i]+1])
            ranges_idxs.append(idxs[range_ends[i-1]+1:range_ends[i]+1])
        else:
            ranges.append(arr[range_ends[i-1]+1:])
            ranges_idxs.append(idxs[range_ends[i-1]+1:])

    means = np.array([np.mean(rng) for rng in ranges])
    grps = group_similar(means, frac_tolerance=frac_tolerance)
    grps_means = [np.mean(means[np.array(grp)]) for grp in grps]
    grps_names = get_group_names(grps_means)

    out = {}
    for grp_mean_idx,name in grps_names:
        grps_idxs = np.sort(grps[grp_mean_idx])

        for i in xrange(len(grps_idxs)):
            out_name = (name + str(i+1)) if len(grps_idxs) > 1 else name
            out[out_name] = ranges_idxs[grps_idxs[i]]

    return out

def clsuter_profile(prof):
    idxs = np.arange(len(prof), dtype=np.int64)

    X = np.array(zip(prof,np.zeros(len(prof))), dtype=np.int)
    bandwidth = estimate_bandwidth(X, quantile=0.3)
    ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    ms.fit(X)
    labels = ms.labels_
    cluster_centers = ms.cluster_centers_
    cluster_centers_1d = cluster_centers[:,0]

    labels_unique = np.unique(labels)
    n_clusters_ = len(labels_unique)
    clusters_ = np.arange(n_clusters_, dtype=np.int)

    if n_clusters_ != 3:
        raise ValueError("Something wrong, the dataset should have only 3 types of images.")

    icflats, icdarks = np.argmax(cluster_centers_1d).astype(np.int64), np.argmin(cluster_centers_1d).astype(np.int64)
    pval = np.delete(cluster_centers_1d, [icflats, icdarks])[0]
    icprojs = np.where(cluster_centers_1d ==pval)[0][0]

    return {'flat': idxs[labels == clusters_[icflats]], \
            'dark': idxs[labels == clusters_[icdarks]], \
            'proj': idxs[labels == clusters_[icprojs]]}

def create_output_dirs(files, split_schema, camera_type, output_folder='Recon', tomo_folder='tomo1'):
    if not len(files):
        raise ValueError('There are no files to read.')

    if not camera_is_available(camera_type):
        raise ValueError('Wrong camera type!')

    input_path = os.path.dirname(os.path.abspath(files[0][0] if camera_type == CAMERA_ANDOR else files[0]))
    root_specimen_dir, spicemen_folder = os.path.split(input_path)

    output_path = os.path.join(root_specimen_dir, output_folder, spicemen_folder, tomo_folder)
    frames_paths = {k: os.path.join(output_path, k) for k in split_schema.keys()}
    frames_paths['proj360'] = os.path.join(output_path, 'proj360')

    for path in frames_paths.values():
        if path is not None:
            if not os.path.exists(path):
                os.makedirs(path)

    return spicemen_folder, frames_paths

def write_data(files, \
               split_schema, \
               camera_type, \
               suffix='%04i', \
               ext='.tif', \
               frac_frames_360deg=-1):
    frame_idxs, frame_idx = {k: 0 for k in split_schema.keys()}, 0

    if not camera_is_available(camera_type):
        raise ValueError('Wrong camera type!')

    spicemen_folder, frames_paths = \
            create_output_dirs(files, split_schema, camera_type)

    for f in files:
        data = None

        if camera_type == CAMERA_PCO_DIMAX:
            data = io.imread(f)
        elif camera_type == CAMERA_ANDOR:
            data = read_andor_data_batch(f)
        else:
            raise ValueError('Wrong camera type!')

        for i,_frame in enumerate(data):
            for name, indices in split_schema.items():
                if frame_idx in indices:
                    path = os.path.join(frames_paths[name], \
                                '_'.join([spicemen_folder, suffix + ext]) % \
                                           frame_idxs[name])
                    io.imsave(path, _frame)
                    print path

                    frame_idxs[name] += 1
                    frame_idx += 1
                    break

    if frac_frames_360deg != -1:
        proj_dir = frames_paths['proj']
        proj360_dir = frames_paths['proj360']

        projs = [f for f in os.listdir(proj_dir) if f.endswith(".tif")]
        projs.sort(key=lambda e: _get_idx(e.split('_')[-1]))
        projs = [os.path.join(proj_dir, f) for f in projs]

        min_val, min_idx = None, -1

        first_proj_data = io.imread(projs[0])
        start_idx = int(len(projs) * frac_frames_360deg)
        sub_projs = projs[-start_idx:]

        for i,path_proj in enumerate(sub_projs):
            proj_data = io.imread(path_proj)
            result = measure.compare_mse(proj_data, first_proj_data)
            print 'MSE of projection #%d: %f' % (len(projs) - start_idx + i, result)
            min_val, min_idx = (result, i) if min_val is None or result < min_val \
                                else (min_val, min_idx)

        offset = len(sub_projs) - min_idx

        proj_start, proj_end = 0,  len(projs) - offset

        print 'Extract projections: %d - %d' % (0, len(projs) - offset)

        for path_proj in projs[proj_start:proj_end+1]:
            print path_proj + ' -> ' + proj360_dir
            shutil.copy(path_proj, proj360_dir)

def split_data(input_sample_dir, camera_type, patch_radius=16, dimax_sep='@', \
               andor_batch_size=100, profile_shrinkage_ratio=50, \
               frac_grp_similarity_tolerance=0.1, frames_fraction_360deg=0.1):
    files = get_data_pathes(input_sample_dir, \
                            camera_type, \
                            sep=dimax_sep, \
                            batch_size=andor_batch_size)
    prof = estimate_profile(files, \
                            camera_type, \
                            patch_radius=patch_radius)
    split_schema = split_profile(prof, \
                                 shrinkage_ratio=profile_shrinkage_ratio, \
                                 frac_tolerance=frac_grp_similarity_tolerance)
    write_data(files, split_schema, camera_type, \
                frac_frames_360deg=frames_fraction_360deg)

def get_sample_paths(input_dir):
    return [os.path.join(input_dir, f) for f in os.listdir(input_dir) \
                if os.path.isdir(os.path.join(input_dir, f))]

def check_camera_type(value):
    cameras = [CAMERA_ANDOR, CAMERA_PCO_DIMAX]
    if value not in cameras:
        raise argparse.ArgumentTypeError("%s is an invalid camera type %s" % (value, str(cameras)))

    return value

class Range(argparse.Action):
    def __init__(self, min=None, max=None, *args, **kwargs):
        self.min = min
        self.max = max
        kwargs["metavar"] = "[%d-%d]" % (self.min, self.max)
        super(Range, self).__init__(*args, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        if not (self.min <= value <= self.max):
            raise argparse.ArgumentError(self, 'invalid choice: %r (choose from [%d-%d])' % (value, self.min, self.max))
        setattr(namespace, self.dest, value)

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("-i", "--input_dir", \
                        help="Path to the folder of the sample", \
                        type=str, \
                        required=True)
    parser.add_argument("-c", "--camera_type", \
                        help="The type of camera used for experiment", \
                        metavar=str([CAMERA_ANDOR, CAMERA_PCO_DIMAX]), \
                        type=check_camera_type, \
                        required=True)
    parser.add_argument("-p", "--patch_radius", \
                        help="The radius of a centered patch to estimate the z-profile", \
                        type=int, \
                        default=16)
    parser.add_argument("-d", "--dimax_sep", \
                        help="The separation symbol of pco dimax files (e.g. raw@00001.tif)", \
                        type=str, \
                        default='@')
    parser.add_argument("-a", "--andor_batch_size", \
                        help="The number of projections to read in single run", \
                        type=int, \
                        default=100)
    parser.add_argument("-s", "--profile_shrinkage_ratio", \
                        help="The ratio to find out whether the peak is significantly bigger than neighbours", \
                        type=int, \
                        default=50)
    parser.add_argument("-t", "--similarity_tolerance", \
                        help="The tolerance coefficient to find almost similar image groups (e.g. flat1 and flat2)", \
                        type=float, \
                        min=0.01, \
                        max=1.0, \
                        action=Range, \
                        default=0.1)
    parser.add_argument("-f", "--fact_frames_360deg", \
                        help="The fraction of projections to find an exact 360 degree projection", \
                        type=float, \
                        min=0.01, \
                        max=1.0, \
                        action=Range, \
                        default=0.1)

    args = parser.parse_args()

    split_data(args.input_dir, \
               args.camera_type, \
               patch_radius=args.patch_radius, \
               dimax_sep=args.dimax_sep, \
               andor_batch_size=args.andor_batch_size, \
               profile_shrinkage_ratio=args.profile_shrinkage_ratio, \
               frac_grp_similarity_tolerance=args.similarity_tolerance, \
               frames_fraction_360deg=args.fact_frames_360deg)

if __name__ == "__main__":
    sys.exit(main())