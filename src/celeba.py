import os
import numpy as np

all_images_path = ""

label_path = "/Volumes/PowerExtension/celebA-labels"
identity_labels = "identity_CelebA.txt"
attribute_labels = "list_attr_celeba.txt"

output_path = "/Volumes/PowerExtension/celebA-labels-output-lists"
feature_name = "Wearing_Lipstick"

image_path = "/Volumes/PowerExtension/img_align_celeba"
select_feature_image_path = "/Volumes/PowerExtension"

#---------------------------------------------------------------------------------------

'''
统计每位名人有多少图片，输出为(人物id, 文件名列表)的哈希表
'''
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
'''
统计各Feature有多少张图片，输出为(特征id, 文件名列表)和(特征名称，特征id)的哈希表
'''
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
输入(特征id, 文件名列表)和(特征名称，特征id)的两个哈希表，输出需要处理的正例图片名称的txt
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

'''
输入图片路径，目标路径，正例图片名称列表路径，被选择的特征名称，输出目标路径与label的npy文件
'''

def generateTrainSetOnFeature(image_path, copy_path, list_path, feature, copy=False):

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
            if copy:
                sourceFile = os.path.join(image_path, imglist[i])
                distFile = os.path.join(copy_path, imglist[i])
                open(distFile, "wb+").write(open(sourceFile, "rb+").read())
            i += 1
            print("No." + str(i) + ' ' + flag)
        print("Positive:" + str(positive))

        while negative < positive:

            if copy:
                sourceFile = os.path.join(image_path, imglist[i])
                distFile = os.path.join(copy_path, imglist[i])
                open(distFile, "wb+").write(open(sourceFile, "rb+").read())
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

def getData(feature_name, select_feature_image_path):

    images_path = os.path.join(select_feature_image_path,feature_name)
    images = os.listdir(images_path)
    images = sorted(images)
    labels_path = os.path.join(images_path, feature_name)
    labels_path = os.path.join(labels_path, feature_name + '.npy')
    labels = np.load(labels_path)
    new_labels = []

    images = images[:-1]
    label_pos = images[-1]
    for i in range(len(images)):
        index = int(images[i].split('.')[0])
        images[i] = os.path.join(images_path, images[i])
        new_labels.append(labels[index-1][0])
    x = np.array(new_labels, dtype=np.int32)
    new_labels = np.ndarray(shape=(len(new_labels),1), buffer=x, dtype=np.int32)
    return images, new_labels

def cAndShuffle(col1, col2):

    a = np.array(col1, dtype=np.int32)
    b = np.array(col2, dtype=np.int32)

    a = np.ndarray(shape=(a.shape[0],1), dtype=np.int32, buffer=a)
    b = np.ndarray(shape=(b.shape[0],1), dtype=np.int32, buffer=b)

    c = np.concatenate((a, b), axis=1)
    np.random.shuffle(c)

    return c[:,0], c[:,1]

def splitData(fnames, labels, factors, base):

    labels.shape = (labels.shape[0],)
    xlabels = labels.tolist()

    test_factor = factors[0]
    val_factor = factors[1]
    train_factor = factors[2]

    test_amount = test_factor * base // 2
    val_amount = val_factor * base // 2
    train_amount = train_factor * base // 2
    
    tests = [0,0]
    vals = [0,0]
    trains = [0,0]

    test_fnames = []
    train_fnames = []
    val_fnames = []

    test_labels = []
    train_labels = []
    val_labels = []

    l_fnames = len(fnames)
    for i in range(l_fnames):
        t = labels[i]
        m = int(t)
        if (tests[m] < test_amount):
            test_fnames.append(i)
            test_labels.append(t)
            tests[m] += 1
        elif (vals[m] < val_amount):
            val_fnames.append(i)
            val_labels.append(t)
            vals[m] += 1
        elif (trains[m] < train_amount):
            train_fnames.append(i)
            train_labels.append(t)
            trains[m] += 1
    test_fnames, test_labels = cAndShuffle(test_fnames, test_labels)
    val_fnames, val_labels = cAndShuffle(val_fnames, val_labels)
    train_fnames, train_labels = cAndShuffle(train_fnames, train_labels)
    real_test_fnames = []
    real_val_fnames = []
    real_train_fnames = []
    for i in test_fnames:
        real_test_fnames.append(fnames[int(i)])
    for i in val_fnames:
        real_val_fnames.append(fnames[int(i)])
    for i in train_fnames:
        real_train_fnames.append(fnames[int(i)])

    return real_test_fnames, test_labels.tolist(), real_val_fnames, val_labels.tolist(), real_train_fnames, train_labels.tolist()

#---------------------------------------------------------------------------------------

#dic_attrib_files, dic_attname_attrib = countFeaturesByFiles(label_path, attribute_labels)
#print(dic_attname_attrib)
#print(len({i:len(dic_attrib_files[i]) for i in dic_attrib_files.keys()}))
#print({i:len(dic_attrib_files[i]) for i in dic_attrib_files.keys()})
#outputSelectedFileNames(dic_attrib_files, dic_attname_attrib, output_path, feature_name)
#generateTrainSetOnFeature(image_path, select_feature_image_path, output_path, feature_name)
#getTrainingData(feature_name, select_feature_image_path)
#outputSelectedFileNames()

    
    




