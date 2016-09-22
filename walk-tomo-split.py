import os
import sys
import argparse
import importlib
import ast
import logging
logger = logging.getLogger('walk-tomo-splitter')
hdlr = logging.FileHandler(os.path.join(os.path.dirname(os.path.realpath(__file__)),'walked-splitting.log'))
formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
hdlr.setFormatter(formatter)
logger.addHandler(hdlr)
logger.setLevel(logging.WARNING)

splitter = importlib.import_module('tomo-split')

#http://stackoverflow.com/questions/2859674/converting-python-list-of-strings-to-their-type
def _tryeval(val):
  try:
    val = ast.literal_eval(val)
  except ValueError:
    pass
  return val

def _list_type(string):
    lst = string.replace('[','')
    lst = lst.replace(']','')
    lst = lst.split(',')
    lst = [k.replace("'", "").replace('"', '').replace(" ", "") for k in lst]
    lst = [_tryeval(l) for l in lst]
    return lst

def get_recon_path(path, root_folder, recon_folder='Recon'):
    comps = []

    while True:
        last, first = os.path.split(path)

        if first == root_folder:
            break

        comps = [first] + comps
        path = last

    return os.path.split(os.path.join(last, first, recon_folder, *comps))[0]

def get_sample_paths(search_dir, sample_names, recon_folder):
    out = []

    for root, dirs, files in os.walk(search_dir):
        if recon_folder not in root and os.path.split(root)[1] in sample_names:
            out.append(root)

    return out

def start_walking(input_dir, camera_type, sample_names, root_folder, \
                  recon_folder='Recon', patch_radius=16, dimax_sep='@', \
                  andor_batch_size=100, profile_shrinkage_ratio=50, \
                  frac_grp_similarity_tolerance=0.1, frames_fraction_360deg=0.1, \
                  logs_path=None):
    sample_paths = get_sample_paths(input_dir, sample_names, recon_folder)
    output_paths = [get_recon_path(p, root_folder, recon_folder=recon_folder) \
                    for p in sample_paths]

    for sample_path, output_path in zip(sample_paths, output_paths):
        splitter.split_data(sample_path, \
                            camera_type, \
                            output_sample_dir=output_path, \
                            patch_radius=patch_radius, \
                            dimax_sep=dimax_sep, \
                            andor_batch_size=andor_batch_size, \
                            profile_shrinkage_ratio=profile_shrinkage_ratio, \
                            frac_grp_similarity_tolerance=frac_grp_similarity_tolerance, \
                            frames_fraction_360deg=frames_fraction_360deg, \
                            logs_path=logs_path)
def main():
    parser = argparse.ArgumentParser('The tomo-splitter which walked away.')

    parser.add_argument("-i", "--input_dir", \
                        help="Path to the folder where to start search", \
                        type=str, \
                        required=True)
    parser.add_argument("-c", "--camera_type", \
                        help="The type of camera used for experiment", \
                        metavar=str([splitter.CAMERA_ANDOR, splitter.CAMERA_PCO_DIMAX]), \
                        type=splitter.check_camera_type, \
                        required=True)
    parser.add_argument("-n", "--sample_names", \
                        help="List of sample names to process", \
                        type=_list_type, \
                        required=True)
    parser.add_argument("-r", "--root_folder", \
                        help="The name of root folder", \
                        type=str, \
                        required=True)
    parser.add_argument("-e", "--recon_folder", \
                        help="The name of reconstruction folder", \
                        type=str, \
                        default='Recon')
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
                        action=splitter.Range, \
                        default=0.1)
    parser.add_argument("-f", "--fact_frames_360deg", \
                        help="The fraction of projections to find an exact 360 degree projection", \
                        type=float, \
                        min=0.01, \
                        max=1.0, \
                        action=splitter.Range, \
                        default=0.1)
    parser.add_argument("-l", "--logs_path", \
                        help="The the path to the logs", \
                        type=str, \
                        default=None)

    args = parser.parse_args()

    start_walking(args.input_dir, \
                  args.camera_type, \
                  args.sample_names, \
                  args.root_folder, \
                  recon_folder=args.recon_folder, \
                  patch_radius=args.patch_radius, \
                  dimax_sep=args.dimax_sep, \
                  andor_batch_size=args.andor_batch_size, \
                  profile_shrinkage_ratio=args.profile_shrinkage_ratio, \
                  frac_grp_similarity_tolerance=args.similarity_tolerance, \
                  frames_fraction_360deg=args.fact_frames_360deg, \
                  logs_path=args.logs_path)

if __name__ == "__main__":
    sys.exit(main())
