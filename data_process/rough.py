
path = 'IMagenet/tiny-imagenet-200/'

def get_id_dictionary():
    id_dict = {}
    for i, line in enumerate(open( path + 'wnids.txt', 'r')):
        id_dict[line.replace('\n', '')] = i
    return id_dict
  
def get_class_to_id_dict():
    id_dict = get_id_dictionary()
    all_classes = {}
    result = {}
    for i, line in enumerate(open(path + 'words.txt', 'r')):
        n_id, word = line.split('\t')[:2]
        all_classes[n_id] = word
    for key, value in id_dict.items():
        result[value] = (key, all_classes[key])      
    return result

def get_data(id_dict):
    print('starting loading data')
    train_data, test_data = [], []
    train_labels, test_labels = [], []
    # test_data = torch.zeros(size=(64,64,3))
    t = time.time()
    for key, value in id_dict.items():
        train_data += [torch.from_numpy(cv2.imread( path + 'train/{}/images/{}_{}.JPEG'.format(key, key, str(i)))) for i in range(500)]
        train_labels_ = np.array([[0]*200]*500)
        train_labels_[:, value] = 1
        train_labels += train_labels_.tolist()
    print(type(train_data), train_data[0].shape)
    for line in open( path + 'val/val_annotations.txt'):
        img_name, class_id = line.split('\t')[:2]
        test_data.append(torch.from_numpy(cv2.imread( path + 'val/images/{}'.format(img_name))))
        test_labels_ = np.array([[0]*200])
        test_labels_[0, id_dict[class_id]] = 1
        test_labels += test_labels_.tolist()

    print('finished loading data, in {} seconds'.format(time.time() - t))
    return torch.stack(train_data).transpose(1,3).transpose(2,3), torch.tensor(train_labels), torch.stack(test_data).transpose(1,3).transpose(2,3), torch.tensor(test_labels)
  
train_data, train_labels, test_data, test_labels = get_data(get_id_dictionary())