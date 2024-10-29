import os
from PIL import Image
import numpy as np
import xml.etree.ElementTree as ET




IMPLEMENTED_DATASETS = {"tuberculosis" : "tuberculosis-phonecamera"}



def get_data_path():
    cwd = os.getcwd()
    dir_to_check = cwd
    for _ in range(4):
        for root, dirs, files in os.walk(dir_to_check):
            if 'data' in dirs:
                return os.path.join(root, 'data')
            if 'dataset' in dirs:
                for root, dirs, files in os.walk(os.path.join(dir_to_check, 'dataset')):
                    if 'data' in dirs:
                        return os.path.join(root, 'data')
                return None
        dir_to_check = os.path.dirname(dir_to_check)
    return None


def get_all_implemented_dataset():
    return list(IMPLEMENTED_DATASETS.keys())

def get_dataset(name):

    data_path = get_data_path()
    if data_path is None:
        assert False, "No data directory found in your repository, please create one under dataset directory"
    os.path.exists(IMPLEMENTED_DATASETS["tuberculosis"]), f"download {name} dataset following DOWNLOAD_DATA.md \n\n {IMPLEMENTED_DATASETS['tuberculosis']} dir was not found"
    path_dataset = os.path.join(data_path, IMPLEMENTED_DATASETS[name])
    if name=="tuberculosis":
        return get_tuberculosis_data(path_dataset)
    assert False, f"No dataset named : {name} implemented"




def get_tuberculosis_data(path_dir):
    path_to_download = path_dir
    def load_jpg(path):
        image = Image.open(path)
        return np.array(image)

    def load_xml(path):
        tree = ET.parse(path)
        root = tree.getroot()
        # Accessing object details

        data = []
        for obj in root.findall("object"):
            label = obj.find("label").text
            pose = obj.find("pose").text
            truncated = obj.find("truncated").text
            occluded = obj.find("occluded").text

            # Access bounding box details
            bndbox = obj.find("bndbox")
            xmin = bndbox.find("xmin").text
            ymin = bndbox.find("ymin").text
            xmax = bndbox.find("xmax").text
            ymax = bndbox.find("ymax").text

            data.append((xmin, xmax, ymin, ymax, label))
        return data

    def get_data(path):
        file_names = os.listdir(path)
        data_jpg = {}
        data_xml = {}
        for file_name in file_names:
            file_path = os.path.join(path, file_name)
            file_without_ext = file_name.split('.')[0]
            if file_name.endswith(".jpg"):
                data_jpg[file_without_ext] = load_jpg(file_path)

            if file_name.endswith(".xml"):
                data_xml[file_without_ext] = load_xml(file_path)
            assert True, "Not supported format found in the directory"

        keys = list(data_jpg.keys())
        x = []
        y = []
        for key in keys:
            x.append(data_jpg[key])
            y.append(data_xml[key])
        return x, y, keys

    x, y, file_name = get_data(path_to_download)
    x = np.array(x)
    return x,y,file_name