import os
import shutil
remove_list = ["model.h5","health_model.h5","dmg_model.h5","dmg_labels.npy"] # "health_labels.npy"
for i in remove_list:
    try:
        os.remove(i)
    except FileNotFoundError:
        pass
shutil.rmtree("screenshot_files", ignore_errors=True)
shutil.rmtree("output_files", ignore_errors=True)
#shutil.rmtree("health_pics", ignore_errors=True)
shutil.rmtree("dmg_pics", ignore_errors=True)
os.mkdir("screenshot_files")
os.mkdir("output_files")
#os.mkdir("health_pics")
os.mkdir("dmg_pics")
print("Clear")