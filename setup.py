import os
import argparse

parser = argparse.ArgumentParser(
                    prog="setup.py",
                    description="Usage of the automated setup",
                    epilog="Text at the bottom of help")

parser.add_argument("-a", "--generate-all", action="store_true", help="reproduce the whole project (matrices, results, thesis)")
parser.add_argument("-m", "--generate-matrices", action="store_true", help="download and create all matrices")
parser.add_argument("-r", "--generate-results", action="store_true", help="reproduce all results (plots/tables)")
parser.add_argument("-t", "--generate-thesis", action="store_true", help="compile the thesis")
parser.add_argument("-s", "--generate-slides", action="store_true", help="compile the slides")
parser.add_argument("-p", "--generate-poster", action="store_true", help="compile the poster")

args = parser.parse_args()

if args.generate_matrices or args.generate_all:
    os.system("python setup/generate_matrices.py")
if args.generate_results or args.generate_all:
    os.system("python setup/generate_results.py")
if args.generate_thesis or args.generate_all:
    os.system("python setup/generate_thesis.py")
if args.generate_slides or args.generate_all:
    os.system("python setup/generate_slides.py")
if args.generate_poster or args.generate_all:
    os.system("python setup/generate_poster.py")
