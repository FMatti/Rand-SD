import __context__

import os
import glob

plot_files = glob.glob("thesis/plots/*.py")
table_files = glob.glob("thesis/tables/*.py")

for plot_file in plot_files:
    os.system("python " + plot_file)

for table_file in table_files:
    os.system("python " + table_file)
