import os
import subprocess
import csv


OPENFACE_DIR = r"D:\Programs\OpenFace\OpenFace_2.0.0_win_x64"
OPENFACE_FEATURE_EXTRACTOR_PROC = "FeatureExtraction.exe"
open_face_bin = os.path.join(OPENFACE_DIR, OPENFACE_FEATURE_EXTRACTOR_PROC)
INDEXES_FILE = r"D:\Programming\pyEye\train_eye\indexes.txt"




def extract_features():
    root_images_dir=r"D:\Programming\pyEye\train_eye\images"
    for file in os.listdir(root_images_dir):
        subprocess.call([open_face_bin, "-f" , os.path.join(root_images_dir, file), "-out_dir", r"D:\Programming\pyEye\train_eye\feature_extraction_openface"])

def indexes_as_dict(indexes_file):
    ret = {}
    with open(indexes_file,'r') as ind_f:
        lines = ind_f.readlines()
        for line in lines:
            name, x, y = line.split(" ")
            ret[name.strip()] = {"x":x.strip(), "y":y.strip()}
    return ret


def unite_csv(csvs_dir, output_filename):
    indexes_dict = indexes_as_dict(INDEXES_FILE)
    csvs_folder = os.path.join(csvs_dir, "csvs")
    files = os.listdir(csvs_folder)
    feature_extracted_dict_list = []
    headers = None
    for file in files:
        to_open_file =os.path.join(csvs_folder, file)
        with open(to_open_file , "r") as csvfile:
            reader = csv.reader(csvfile)
            headers = next(reader,None)
            for row in reader:
                try:
                    new_row = row
                    jpg_name = file[:-4]+".jpg"
                    print(jpg_name)
                    new_row.append(jpg_name)
                    new_row.append(indexes_dict[jpg_name]['y'])
                    new_row.append(indexes_dict[jpg_name]['x'])
                    feature_extracted_dict_list.append(row)
                except Exception as e:
                    print("Failed at : {0}".format(file))
    with open(output_filename, "w", newline='') as outfile:
        headers=headers + ['file_name', 'expected_x', 'expected_y']
        writer = csv.writer(outfile, delimiter=',')
        writer.writerow(headers)
        for row in feature_extracted_dict_list:
            writer.writerow(row)





if __name__=="__main__":
    unite_csv(r"D:\Programming\pyEye\train_eye\feature_extraction_openface", r"D:\Programming\pyEye\train_eye\feature_extraction_openface\joint_csv.csv")
