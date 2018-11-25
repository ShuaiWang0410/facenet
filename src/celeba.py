import os
import numpy as np

all_images_path = ""

label_path = "/Volumes/新加卷/celebA-labels"
identity_labels = "identity_CelebA.txt"
attribute_labels = "list_attr_celeba.txt"

output_path = "/Volumes/新加卷/celebA-labels-output-lists"
feature_name = "Wearing_Lipstick"

image_path = "/Volumes/新加卷/img_align_celeba"
select_feature_image_path = "/Volumes/新加卷"

#---------------------------------------------------------------------------------------

def countIdentities(label_path, identity_labels):

    identity_path = os.path.join(label_path, identity_labels)

    dic_ids_file = {}

    with open(identity_path) as file:
        s = file.readline()

        while s:
            celeb_id, celeb_file = s.split()
            if not dic_ids_file.has_key(celeb_id):
                dic_ids_file[celeb_id] = []
            dic_ids_file[celeb_id].append(celeb_file)
            s = file.readline()
    
    return dic_ids_file

def countFeaturesByFiles(label_path, attribute_labels):
    
    attribute_path = os.path.join(label_path, attribute_labels)
    
    dic_attname_attrib = {}
    dic_attrib_files = {}
    l_filenames = 0
    cnt_filenames = 0

    with open(attribute_path) as file:
        l_filenames += int(file.readline())
        attributes = file.readline().split()
        for i in range(len(attributes)):
            dic_attname_attrib[attributes[i]] = i

        s = file.readline()

        while s:
            cnt_filenames += 1
            si = s.split()
            filename = si[0]
            atts = si[1:]
            for i in range(len(atts)):
                if (atts[i] == '1'):
                    attr_id = i
                    if attr_id not in dic_attrib_files.keys():
                        dic_attrib_files[attr_id] = []
                    dic_attrib_files[attr_id].append(filename)
            s = file.readline()

        assert(cnt_filenames == l_filenames)

    return dic_attrib_files, dic_attname_attrib

'''
def countFileByFeatures(label_path, attribute_labels):
'''

def outputSelectedFileNames(dic_attrib_files, dic_attname_attrib, output_path, output_file):

   try:
       os.mkdir(output_path)
   except FileExistsError:
       print(output_path + " already exists")

   filename_list = dic_attrib_files[dic_attname_attrib[output_file]]
   filename_list = sorted(filename_list)
   output_file += ".txt"
   output_file_path = os.path.join(output_path, output_file)
   count = 0
   with open(output_file_path,"w+") as out_file:

       for i in filename_list:
           out_file.write(i + "\n")
           count += 1
           print(count)

def generateTrainSetOnFeature(image_path, copy_path, list_path, feature):

    featurelist_path = os.path.join(list_path, feature + ".txt")

    copy_path = os.path.join(copy_path, feature)
    try:
        os.mkdir(copy_path)
    except FileExistsError:
        print(copy_path + " exists!")

    negative = 0
    positive = 0
    labels = []

    with open(featurelist_path) as file:

        imglist = os.listdir(image_path)
        imglist = sorted(imglist)
        s = file.readline()
        i = 0
        flag = ''
        while s:
            t = s[:-1]
            if t != imglist[i]:
                negative += 1
                labels.append(0.)
                flag = 'N'
            else:
                positive += 1
                s = file.readline()
                labels.append(1.)
                flag = 'P'
            #sourceFile = os.path.join(image_path, imglist[i])
            #distFile = os.path.join(copy_path, imglist[i])
            #open(distFile, "wb+").write(open(sourceFile, "rb+").read())
            i += 1
            print("No." + str(i) + ' ' + flag)
        print("Positive:" + str(positive))

        while negative < positive:

            #sourceFile = os.path.join(image_path, imglist[i])
            #distFile = os.path.join(copy_path, imglist[i])
            #open(distFile, "wb+").write(open(sourceFile, "rb+").read())
            labels.append(0.)
            i += 1
            negative += 1
        print("Negative:" + str(negative))


    feature_label_path = os.path.join(copy_path, feature)
    try:
        os.mkdir(feature_label_path)
    except FileExistsError:
        print(feature_label_path + " exists")
    feature_label_path = os.path.join(feature_label_path, feature + ".npy")
    x = np.array(labels, dtype=np.int32)
    labels = np.ndarray(shape =(x.shape[0],1), buffer=x, dtype=np.int32)
    np.save(file=feature_label_path, arr=labels)

def getTrainingData(feature_name, select_feature_image_path):

    images_path = os.path.join(select_feature_image_path,feature_name)
    images = os.listdir(images_path)
    images = sorted(images)
    images = images[:-1]
    for i in range(len(images)):
        images[i] = os.path.join(images_path, images[i])
    labels_path = os.path.join(images_path, feature_name)
    labels_path = os.path.join(labels_path, feature_name + '.npy')
    labels = np.load(labels_path)

    return images, labels



#---------------------------------------------------------------------------------------

#dic_attrib_files, dic_attname_attrib = countFeaturesByFiles(label_path, attribute_labels)
#print(dic_attname_attrib)
#print(len({i:len(dic_attrib_files[i]) for i in dic_attrib_files.keys()}))
#print({i:len(dic_attrib_files[i]) for i in dic_attrib_files.keys()})
#outputSelectedFileNames(dic_attrib_files, dic_attname_attrib, output_path, feature_name)
#generateTrainSetOnFeature(image_path, select_feature_image_path, output_path, feature_name)
getTrainingData(feature_name, select_feature_image_path)
#outputSelectedFileNames()

    
    




