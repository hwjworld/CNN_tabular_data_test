from PIL import Image
import os
directory = "/Users/jeromehuang/data/hwjmy/2022uq/courses/2022s2/ai_internship_2022s2/project/dataset/Art Images/dataset/dataset_updated/validation_set"

classes = sorted(
    entry.name for entry in os.scandir(directory) if entry.is_dir())

# class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
print(classes)
for c in classes:
    path = directory+"/"+c
    filenames = sorted( entry.name for entry in os.scandir(path) if
                       entry.is_file())
    for fn in filenames:
        with open(path+"/"+fn, "rb") as f:
            try:
                img = Image.open(f)
                # print(img)
            except:
                os.remove(path+"/"+fn)
                print("removed:"+path+"/"+fn)



# with open(path, "rb") as f:
#     img = Image.open(f)
#     img.convert("RGB")

