import os
import shutil
remove_list = ["model.h5"]
for i in remove_list:
    try:
        os.remove(i)
    except FileNotFoundError:
        pass
shutil.rmtree("screenshot_files", ignore_errors=True)
shutil.rmtree("output_files", ignore_errors=True)
os.mkdir("screenshot_files")
os.mkdir("output_files")
print("Clear")