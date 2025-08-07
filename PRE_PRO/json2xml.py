import sys
import os
import json
import shutil
import time

sys.path.append(r'../')
from config import config


def convert(xml_template_path, annotation_path, output_path):
    f_template = open(xml_template_path, 'r')
    f_target = open(annotation_path, 'r')
    f_output = open(output_path, 'w+')
    print("annotation:" + annotation_path)
    print("output:" + output_path)

    content_template = ""
    output_xml = ""

    for l in f_target:
        output_xml += l

    parsed_json = json.loads(output_xml)

    anno_no = 0  # annotations index number
    coord_no = 0  # reset in each annotation

    for anno in parsed_json['annotation']:
    
        if anno['type'] == "freehand":
            anno_type = "Polygon"
        else:
            anno_type = anno['type']

        if anno['partOfGroup'] not in ['HCC','tumor', 'Result']:
            anno_partOfGroup = "else"
        else:
            anno_partOfGroup = "tumor"

        anno_tag = '\t\t<Annotation Name="' + str(
            anno_no) + '" Type="' + anno_type + '" PartOfGroup="' + anno_partOfGroup + '" Color="#F4FA58">\n'

        coord_content = "\t\t\t<Coordinates>\n"
        for c in anno['coordinates']:
            coord_content += '\t\t\t\t<Coordinate Order="' + str(coord_no) + '" X="' + str(c['x']) + '" Y="' + str(
                c['y']) + '" />\n'
            coord_no += 1
        coord_content += "\t\t\t</Coordinates>\n"
        coord_no = 0
        content_template += anno_tag + coord_content + "\t\t</Annotation>\n"
        anno_no += 1

    body_template = ""
    for l in f_template:
        if "<Annotations>" in l:
            body_template += l
            body_template += content_template
        else:
            body_template += l

    f_output.write(body_template)


def check_file(annotation_path, processed_path, copyto_target):
    if not os.path.isfile(annotation_path):
        print('[ERROR] RAW DATA annotation not founded: ' + uuid)
        exit()

    if not os.path.isdir(f'{processed_path}/{uuid}'):
        print("[INFO] Doesn't exist path, make directory:")
        print(f'{processed_path}/{uuid}')
        os.makedirs(f'{processed_path}/{uuid}')
        print("[INFO] Directory made!")

    try:
        shutil.copyfile(annotation_path, copyto_target)
    except:
        print('[ERROR] ' + uuid, sys.exc_info()[0])
        exit()


if __name__ == '__main__':
    uuid = sys.argv[1]
    processed_path = config["preprocess_save_path"]
    annotation_path = config["annotation_json_path"] + f"/{uuid}/annotation.json"
    xml_template_path = "./template.xml"
    output_path = f"{processed_path}/{uuid}/{uuid}_output.xml"
    copyto_target = f"{processed_path}/{uuid}/{uuid}_annotation.json"
    check_file(annotation_path, processed_path, copyto_target)
    convert(xml_template_path, annotation_path, output_path)
    print("[INFO] Converted Annotations Done: " + uuid)
