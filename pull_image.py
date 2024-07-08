import os
import shutil
import numpy as np
import pandas as pd
from PIL import Image
from PIL import Image, ImageOps
# Example dataframe

df = pd.read_csv("./tracked_targets.csv")
def round_up_to_even(number):
    return int(np.ceil(number / 2.0) * 2)
def get_largest_box(df):
    result = {}
    for id_, group in df.groupby('obj_id'):
        max_width = (group['x2'] - group['x1']).max()
        max_height = (group['y2'] - group['y1']).max()
        
        # Round up to the nearest even number
        max_width = round_up_to_even(max_width)
        max_height = round_up_to_even(max_height)
        
        result[id_] = (max_width, max_height)
    return result

def enlarge_box(box, factor=2):
    width, height = box
    return int(width * factor), int(height * factor)

def ensure_folder_exists(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
def pad_image(image, left, top, right, bottom, color=0):
    return ImageOps.expand(image, (int(left), int(top), int(right), int(bottom)), fill=color)
def get_images(df, largest_boxes):
    i = 0
    for _, row in df.iterrows():
        image_path = os.path.join('./imageset_2_10_100', f'freqsweep{str(int(row["frame_number"])).zfill(4)}.png')
        image = Image.open(image_path)
        
        width, height = largest_boxes[row['obj_id']]
        #ratio = width/height
        width_even = (width%2 == 0)
        height_even = (height%2 == 0)

        if width_even != height_even:
            width += 1


        enlarged_width, enlarged_height = enlarge_box((width, height))
        
        x_center = (row['x1'] + row['x2']) // 2
        y_center = (row['y1'] + row['y2']) // 2
        
        x1 = int(x_center - enlarged_width // 2)
        y1 = int(y_center - enlarged_height // 2)
        x2 = int(x_center + enlarged_width // 2)
        y2 = int(y_center + enlarged_height // 2)

        left_pad = max(0, -x1)
        top_pad = max(0, -y1)
        right_pad = max(0, x2 - image.width)
        bottom_pad = max(0, y2 - image.height)

        x1 = max(x1, 0)
        y1 = max(y1, 0)
        x2 = min(x2, image.width)
        y2 = min(y2, image.height)
        
        cropped_image = image.crop((x1, y1, x2, y2))
        if left_pad > 0 or top_pad > 0 or right_pad > 0 or bottom_pad > 0:
            cropped_image = pad_image(cropped_image, left_pad, top_pad, right_pad, bottom_pad)
        
        frame_number = str(int(i)).zfill(4)
        i += 1
        obj_id = int(row['obj_id'])

        save_path = os.path.join('./obj', f'obj{obj_id}', f'cropped_freqsweep{frame_number}.png')
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        cropped_image.save(save_path)
        print(f"Cropped image saved as {save_path}")


largest_boxes = get_largest_box(df)
#cut_images(df, largest_boxes)


particle_id = 32588


selected_dataframe = df[df["obj_id"] == particle_id]

#given by width, height
dims = get_largest_box(selected_dataframe)
print(dims)
largest_bounding_box = enlarge_box(get_largest_box(selected_dataframe)[particle_id])

save_location = "./obj/obj" + str(particle_id) +"/"

ensure_folder_exists(save_location)

get_images(selected_dataframe, dims)

