import numpy as np
import scipy.io as sio
import h5py
if __name__ == '__main__':
    # # MIRFlickr
    # path = '/data/sunyu/170/Datasets/MIRFlikr/Process/20015/partition_for_incremental/'
    path = '/data/sunyu/170/Datasets/MIRFlikr/Process/20015/partition_for_incremental/random/'
    query_image_path = path + "mir_2400_query_image.mat"
    query_label_path = path + "mir_2400_query_label.mat"
    query_tag_path = path + "mir_2400_query_tag.mat"
    retrieval_image_path = path + "mir_17615_retrieval_image.mat"
    retrieval_label_path = path + "mir_17615_retrieval_label.mat"
    retrieval_tag_path = path + "mir_17615_retrieval_tag.mat"

    mir_2400_query_image = h5py.File(query_image_path)['mir_2400_query_image'][:]
    mir_2400_query_image = mir_2400_query_image.transpose(3,2,1,0)
    mir_2400_query_label = h5py.File(query_label_path)['mir_2400_query_label'][:]
    mir_2400_query_label = mir_2400_query_label.transpose(1,0)
    mir_2400_query_tag = h5py.File(query_tag_path)['mir_2400_query_tag'][:]
    mir_2400_query_tag = mir_2400_query_tag.transpose(1,0)
    mir_17615_retrieval_image = h5py.File(retrieval_image_path)['mir_17615_retrieval_image'][:]
    mir_17615_retrieval_image = mir_17615_retrieval_image.transpose(3,2,1,0)
    mir_17615_retrieval_label = h5py.File(retrieval_label_path)['mir_17615_retrieval_label'][:]
    mir_17615_retrieval_label = mir_17615_retrieval_label.transpose(1,0)
    mir_17615_retrieval_tag = h5py.File(retrieval_tag_path)['mir_17615_retrieval_tag'][:]
    mir_17615_retrieval_tag = mir_17615_retrieval_tag.transpose(1,0)
    print(mir_2400_query_image.shape)
    print(mir_2400_query_label.shape)
    print(mir_2400_query_tag.shape)
    print(mir_17615_retrieval_image.shape)
    print(mir_17615_retrieval_label.shape)
    print(mir_17615_retrieval_tag.shape)
    # np.save('./MIRFlickr/mir_2400_query_image.npy',mir_2400_query_image)
    # np.save('./MIRFlickr/mir_2400_query_label.npy',mir_2400_query_label)
    # np.save('./MIRFlickr/mir_2400_query_tag.npy',mir_2400_query_tag)
    # np.save('./MIRFlickr/mir_17615_retrieval_image.npy',mir_17615_retrieval_image)
    # np.save('./MIRFlickr/mir_17615_retrieval_label.npy',mir_17615_retrieval_label)
    # np.save('./MIRFlickr/mir_17615_retrieval_tag.npy',mir_17615_retrieval_tag)
    np.save('./MIRFlikcr_random/mir_2400_query_image.npy',mir_2400_query_image)
    np.save('./MIRFlikcr_random/mir_2400_query_label.npy',mir_2400_query_label)
    np.save('./MIRFlikcr_random/mir_2400_query_tag.npy',mir_2400_query_tag)
    np.save('./MIRFlikcr_random/mir_17615_retrieval_image.npy',mir_17615_retrieval_image)
    np.save('./MIRFlikcr_random/mir_17615_retrieval_label.npy',mir_17615_retrieval_label)
    np.save('./MIRFlikcr_random/mir_17615_retrieval_tag.npy',mir_17615_retrieval_tag)

    # NUS-WIDE
    # path = '/data/sunyu/170/Datasets/NUS-WIDE/Process/195834_top21/partition_for_incremental/'
    # query_image_path = path + "nus_2100_query_image.mat"
    # query_label_path = path + "nus_2100_query_label.mat"
    # query_tag_path = path + "nus_2100_query_tag.mat"
    # retrieval_image_path = path + "nus_193734_retrieval_image.mat"
    # retrieval_label_path = path + "nus_193734_retrieval_label.mat"
    # retrieval_tag_path = path + "nus_193734_retrieval_tag.mat"
    #
    # nus_2100_query_image = h5py.File(query_image_path)['nus_2100_query_image'][:]
    # nus_2100_query_image = nus_2100_query_image.transpose(3,2,1,0)
    # nus_2100_query_label = h5py.File(query_label_path)['nus_2100_query_label'][:]
    # nus_2100_query_label = nus_2100_query_label.transpose(1,0)
    # nus_2100_query_tag = h5py.File(query_tag_path)['nus_2100_query_tag'][:]
    # nus_2100_query_tag = nus_2100_query_tag.transpose(1,0)
    # nus_193734_retrieval_image = h5py.File(retrieval_image_path)['nus_193734_retrieval_image'][:]
    # nus_193734_retrieval_image = nus_193734_retrieval_image.transpose(3,2,1,0)
    # nus_193734_retrieval_label = h5py.File(retrieval_label_path)['nus_193734_retrieval_label'][:]
    # nus_193734_retrieval_label = nus_193734_retrieval_label.transpose(1,0)
    # nus_193734_retrieval_tag = h5py.File(retrieval_tag_path)['nus_193734_retrieval_tag'][:]
    # nus_193734_retrieval_tag = nus_193734_retrieval_tag.transpose(1,0)
    # print(nus_2100_query_image.shape)
    # print(nus_2100_query_label.shape)
    # print(nus_2100_query_tag.shape)
    # print(nus_193734_retrieval_image.shape)
    # print(nus_193734_retrieval_label.shape)
    # print(nus_193734_retrieval_tag.shape)
    # np.save('./NUSWIDE/nus_2100_query_image.npy',nus_2100_query_image)
    # np.save('./NUSWIDE/nus_2100_query_label.npy',nus_2100_query_label)
    # np.save('./NUSWIDE/nus_2100_query_tag.npy',nus_2100_query_tag)
    # np.save('./NUSWIDE/nus_193734_retrieval_image.npy',nus_193734_retrieval_image)
    # np.save('./NUSWIDE/nus_193734_retrieval_label.npy',nus_193734_retrieval_label)
    # np.save('./NUSWIDE/nus_193734_retrieval_tag.npy',nus_193734_retrieval_tag)

