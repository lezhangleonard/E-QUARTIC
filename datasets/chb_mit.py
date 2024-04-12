import mne
import os
import torch
from torch.utils.data import Dataset, DataLoader

class CHBMitDataset(Dataset):
    def __init__(self, data_path, patient_folders, window_size=4):
        self.data_path = data_path
        self.window_size = window_size
        self.data = []
        self.labels = []

        # Load all the raw files for each patient
        raws = []
        for folder_name in patient_folders:

        # for folder_name in os.listdir(data_path):
        #     if folder_name.startswith("chb") and folder_name[3:].isdigit():
            raw_files = []
            patient_folder_path = os.path.join(data_path, folder_name)
            for file_name in os.listdir(patient_folder_path):
                if file_name.endswith(".edf") and file_name.startswith("chb"):
                    file_path = os.path.join(patient_folder_path, file_name)
                    print("Reading: {}".format(file_path))
                    raw_file = mne.io.read_raw_edf(file_path, preload=True, verbose='error')
                    raw_files.append(raw_file)
            if raw_files:
                raw = mne.concatenate_raws(raw_files)
                raws.append(raw)

        # Loop through each patient's raw object
        for patient_id, raw in enumerate(raws):
            # Load the summary file for the patient
            folder_name = "chb{:02d}".format(patient_id + 1)
            summary_file_path = os.path.join(data_path, folder_name, "{}-summary.txt".format(folder_name))
            with open(summary_file_path, "r") as f:
                summary_text = f.read()

            # Loop through each line in the summary file
            for line in summary_text.split("\n"):
                if line.startswith("Seizure"):
                    # Seizure case
                    fields = line.split()
                    start_time = float(fields[1])
                    end_time = float(fields[2])
                    self.load_data(raw, start_time, end_time, seizure=True)
                else:
                    # No-seizure case
                    self.load_data(raw, seizure=False)

    def load_data(self, raw, start_time=None, end_time=None, seizure=False):
        print("  - loading data...")
        if seizure:
            # Crop the raw data to the seizure interval
            raw_crop = raw.copy().crop(tmin=start_time, tmax=end_time)
        else:
            raw_crop = raw.copy()

        # Extract the sampling rate
        sfreq = raw_crop.info["sfreq"]

        # Calculate the number of samples in the window
        window_samples = int(self.window_size * sfreq)

        # Get the total number of samples in the raw data
        n_samples = raw_crop.n_times

        # Calculate the number of windows
        n_windows = int(n_samples / window_samples)

        # Loop through each window
        for j in range(n_windows):
            # Calculate the start and end indices of the window
            start_index = int(j * window_samples)
            end_index = int((j + 1) * window_samples)

            # Extract the data for the window
            window_data = raw_crop.get_data(start=start_index, stop=end_index)

            # Append the window data and label to the data and labels lists
            self.data.append(window_data)
            self.labels.append(int(seizure))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return torch.from_numpy(self.data[index]), self.labels[index]

# Example usage:
data_path = "../../../../chb-mit-scalp-eeg-database-1.0.0"
patient_folders = ["chb01"]
window_size = 4

dataset = CHBMitDataset(data_path, patient_folders, window_size=window_size)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Iterate over the dataloader
for data, labels in dataloader:
    print(data, labels)
    input()






# import mne
# import os
# import torch
# from torch.utils.data import Dataset, DataLoader
# # import warnings
# # warnings.filterwarnings("ignore")

# class CHBMitDataset(Dataset):
#     def __init__(self, data_path, window_size=4):
#         self.data_path = data_path
#         self.window_size = window_size
#         self.data = []
#         self.labels = []

#         # loop through all the patient folders in the data path
#         for folder_name in os.listdir(data_path):

#             # check if the folder name corresponds to a patient folder
#             if folder_name.startswith("chb") and folder_name[3:].isdigit():

#                 # extract the patient ID from the folder name
#                 patient_id = int(folder_name[3:])

#                 # load the summary file for the patient
#                 summary_file_path = os.path.join(data_path, folder_name, "{}-summary.txt".format(folder_name))
#                 with open(summary_file_path, "r") as f:
#                     summary_text = f.read()

#                 # loop through each line in the summary file
#                 for line in summary_text.split("\n"):
#                     if line.startswith("Seizure"):
#                         # Seizure case
#                         fields = line.split()
#                         start_time = float(fields[1])
#                         end_time = float(fields[2])
#                         self.load_data(folder_name, patient_id, start_time, end_time, seizure=True)
#                     else:
#                         # No-seizure case
#                         self.load_data(folder_name, patient_id, seizure=False)

#     def load_data(self, folder_name, patient_id, start_time=None, end_time=None, seizure=False):
#         print(folder_name, patient_id)
#         for edf_file in os.listdir(os.path.join(self.data_path, folder_name)):
#             if(edf_file[-1] == 'f'):
#                 # load the EEG data for the patient using MNE
#                 eeg_file_path = os.path.join(self.data_path, folder_name, edf_file)

#                 raw = mne.io.read_raw_edf(eeg_file_path, preload=True, verbose='error')

#                 if seizure:
#                     # crop the EEG data to the seizure interval
#                     raw_crop = raw.crop(tmin=start_time, tmax=end_time)
#                 else:
#                     raw_crop = raw

#                 # extract the sampling rate
#                 sfreq = raw_crop.info["sfreq"]

#                 # calculate the number of samples in the window
#                 window_samples = int(self.window_size * sfreq)

#                 # get the total number of samples in the raw data
#                 n_samples = raw_crop.n_times

#                 # calculate the number of windows
#                 n_windows = int(n_samples / window_samples)

#                 # loop through each window
#                 for j in range(n_windows):
#                     # calculate the start and end indices of the window
#                     start_index = int(j * window_samples)
#                     end_index = int((j + 1) * window_samples)

#                     # extract the data for the window
#                     window_data = raw_crop.get_data(start=start_index, stop=end_index)

#                     # append the window data and label to the data and labels lists
#                     self.data.append(window_data)
#                     self.labels.append(int(seizure))

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, index):
#         return torch.from_numpy(self.data[index]), self.labels[index]

# # Example usage:
# data_path = "../../../../chb-mit-scalp-eeg-database-1.0.0"
# window_size = 4

# dataset = CHBMitDataset(data_path, window_size=window_size)
# dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# # Iterate over the dataloader
# for data, labels in dataloader:
#     print(data, labels)
#     input()
#     # Your



































































# import os
# import mne #import hdf5storage as h5
# import torch
# from torch.utils.data import DataLoader, Dataset
# import torchvision.transforms as tvt
# import numpy as np

# from imblearn.over_sampling import SMOTE
# from imblearn.under_sampling import RandomUnderSampler
# from imblearn.pipeline import Pipeline

# from scipy.signal import butter, lfilter, filtfilt

# import matplotlib.pyplot as plt

# lowcut, highcut, fs = 1, 30, 256

# # def butter_bandpass(lowcut, highcut, fs, order=5):
# #     return butter(order, [lowcut, highcut], fs=fs, btype='band')

# # def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
# #     b, a = butter_bandpass(lowcut, highcut, fs, order=order)
# #     y = filtfilt(b, a, data)
# #     return y





# class CHBMIT(Dataset):
#     def __init__(self, data_folder, patients_list, window_size_sec, augment=False):
        
#         assert os.path.exists(data_folder), 'Data folder does not exist'
#         self.sec_per_hour = 3600
#         self.samples_per_sec = 256
#         self.samples_per_hour = self.samples_per_sec * self.sec_per_hour
        
#         self.data_folder = os.path.abspath(data_folder)
#         self.patient_folders = patients_list
#         self.window_size_sec = window_size_sec
#         self.windows_per_file = self.sec_per_hour / self.window_size_sec
#         self.samples_per_window = self.window_size_sec * self.samples_per_sec
#         self.edf_files = []
#         self.patients_seizures_windows_indexes = []
#         self.windows_per_patient = []
#         self.data = []
#         self.labels = []

#         #DETERMINE DATASET LENGHT BASED ON DESIRED NUMBER OF PATIENTS AND NUMBER OF RECORDINGS PER PATIENT
#         # tot_hours = 0
#         # for patient_num, patient in enumerate(self.patient_folders):
#         #   patient_directory = os.path.join(self.data_folder, patient)
#         #   self.patients_seizures_windows_indexes.append([])
#         #   self.edf_files.append([])
            
#         #   # GET SEIZURE INDEXES FROM ALL FILES OF CURRENT PATIENT ##########################
#         #   summary_file = open(os.path.join(patient_directory, patient+'-summary.txt'), 'r')
#         #   # print(os.path.join(patient_directory, patient+'-summary.txt'))
#         #   lines = summary_file.readlines()
#         #   current_base_window = 0
#         #   ROW = 0 
#         #   while ROW < len(lines):
#         #       if ("File Name" in lines[ROW]):
#         #           filename = lines[ROW][-13:-1]
#         #           ROW = ROW + 3
#         #           seizure_num = int(lines[ROW][-2])
#         #           if (seizure_num == 0):
#         #               ROW = ROW + 2
#         #           else:
#         #               for i in range(seizure_num):
#         #                   ROW = ROW + 1
#         #                   start = int(lines[ROW][22:-9])
#         #                   start_window = current_base_window + int(start / self.window_size_sec)
#         #                   ROW = ROW + 1
#         #                   end = int(lines[ROW][19:-9])
#         #                   end_window = current_base_window + int(end / self.window_size_sec)
#         #                   self.patients_seizures_windows_indexes[patient_num].append([start_window, end_window])
#         #               ROW = ROW + 2
#         #           current_base_window = current_base_window + self.windows_per_file
#         #       else:
#         #           ROW = ROW + 1
#         #   ################################################################################

#         print("Loading patient data...")
#         for patient_num, patient in enumerate(self.patient_folders):
#             patient_directory = os.path.join(self.data_folder, patient)
#             summary_file = open(os.path.join(patient_directory, patient+'-summary.txt'), 'r')
#             lines = summary_file.readlines()
#             current_base_window = 0
#             ROW = 0 
#             while ROW < len(lines):
#                 if ("File Name" in lines[ROW]):
#                     filename = lines[ROW][11:-1]
#                     print("  - Patient ", filename)
#                     ROW = ROW + 3
#                     seizure_num = int(lines[ROW][-2])
#                     if (seizure_num == 0):
#                         ROW = ROW + 2
#                         start_window = -1
#                         end_window = -1
#                     else:
#                         for i in range(seizure_num):
#                             ROW = ROW + 1
#                             start = int(lines[ROW][22:-9])
#                             start_window = current_base_window + int(start / self.window_size_sec)
#                             ROW = ROW + 1
#                             end = int(lines[ROW][19:-9])
#                             end_window = current_base_window + int(end / self.window_size_sec)
#                         ROW = ROW + 2

#                     data_file = mne.io.read_raw_edf( os.path.join(patient_directory, filename), verbose='error')
#                     # data_file.load_data()
#                     # data_file.plot()
#                     # data_file.filter(1,40) 
#                     # data = mne.filter.filter_data(data_file.get_data(), fs, l_freq=lowcut, h_freq=highcut, method='iir', verbose='error')
#                     # data_file.plot()
                    
#                     data = data_file.get_data()
#                     # data = np.mean(data, axis = 0)
#                     Wstart = 0
#                     Wstep = self.window_size_sec * self.samples_per_sec
#                     Wend = Wstep
#                     w = 0

#                     # while(Wend < data_file.shape[0]):
#                     #   self.data.append(data_file[Wstart:Wend])
#                     #   if(start_window <= w and end_window >= w):
#                     #       self.labels.append(1)
#                     #   else:
#                     #       self.labels.append(0)
#                     #   Wstart = Wstart + Wstep
#                     #   Wend = Wend + Wstep
#                     #   w = w + 1


#                     if(data.shape[0] >=23):
#                         # windowed_data = np.array_split(data, data.shape[0], axis=1)
#                         # self.data.append( [sub_arr for sub_arr in windowed_data])
#                         self.data.append(data[0:23, Wstart:Wend])
#                         # print(type(data), data.shape)
#                         # print(self.data)
#                         # input()
#                         while(Wend < data.shape[1]):
#                             if(start_window <= w and end_window >= w):
#                                 self.labels.append(1)
#                             else:
#                                 self.labels.append(0)
#                             Wstart = Wstart + Wstep
#                             Wend = Wend + Wstep
#                             w = w + 1
#                 else:
#                     ROW = ROW + 1

#         self.data = np.array(self.data)
#         self.labels = np.array(self.labels)

#         neg = len([x for x in self.labels if x == 0])
#         pos = len([x for x in self.labels if x == 1])
#         print("Tot number of windows: ", int(len(self.labels)) )
#         print("Positive Windows: {} ({:.2f}%)".format(pos, 100*pos/len(self.labels)))
#         print("Negative Windows: {} ({:.2f}%)\n".format(neg, 100*neg/len(self.labels)))
#         if(augment):
#             # self.augment_positive_samples(pos, neg)
#             self.data_augmentation()
#         # x = range(self.data.shape[2])
#         # plt.figure
#         # plt.plot(x, self.data[0][0])
#         # self.data = butter_bandpass_filter(self.data, lowcut, highcut, fs, order=3)
#         # plt.plot(x, self.data[0][0])
#         # plt.show()
#         # input()
#         # print(self.data.shape)
#         # input()

#             ################################################################################

#         #   files = sorted(os.listdir(patient_directory))
#         #   patient_hours = 0
#         #   for file in files:
#         #       if file[-4:] == ".edf":
#         #           self.edf_files[patient_num].append( os.path.join(patient_directory, file) )
#         #           tot_hours = tot_hours + 1
#         #           patient_hours = patient_hours + 1
#         #   self.windows_per_patient.append( int((patient_hours * self.samples_per_hour) / (self.window_size_sec * self.samples_per_sec) ) )
#         # samples = tot_hours * self.samples_per_hour
#         # self.tot_windows = int(samples / (self.window_size_sec * self.samples_per_sec))


#         # print("Samples per hour: ", self.samples_per_hour)
#         # print("Window size [s]: ", self.window_size_sec)
#         # print("Data path: ", self.data_folder)
#         # print("Number of patients: ", len(self.edf_files))
#         # print("Patients: ", self.patient_folders)
#         # for p, patient in enumerate(self.patient_folders):
#         #   print("   - Patient: ", patient)
#         #   for f in self.edf_files[p]:
#         #       print("      - {}".format(f))
#         #   print("   - Patient Windows: {}".format(self.windows_per_patient[p]))
#         #   print("   - Windows having seizures: {}".format(self.patients_seizures_windows_indexes[p]))

#         # print("Total number of windows: ", self.tot_windows)
#         # input()

#         ##################################################################################################

#     def data_augmentation(self):
#         over = SMOTE(sampling_strategy=1.0)
#         under = RandomUnderSampler(sampling_strategy=0.0)
#         steps = [('o', over)]#, ('u', under)]
#         pipeline = Pipeline(steps=steps)
#         tmp_data = np.reshape(self.data, (self.data.shape[0],-1))
#         # self.data, self.labels = sm.fit_resample(self.data, self.labels)
#         new_data, new_targets = pipeline.fit_resample(tmp_data, self.labels)
#         self.data = np.reshape(new_data, (new_data.shape[0],23,-1))
#         self.labels = new_targets
#         neg = len([x for x in self.labels if x == 0])
#         pos = len([x for x in self.labels if x == 1])
#         print("Augmented Tot number of windows: ", int(len(self.labels)) )
#         print("Augmented Positive Windows: {} ({:.2f}%)".format(pos, 100*pos/len(self.labels)))
#         print("Augmented Negative Windows: {} ({:.2f}%)\n".format(neg, 100*neg/len(self.labels)))



#     def augment_positive_samples(self, n_positive, n_negative):
#         replicate_factor = int(n_negative/n_positive)
#         # print("replicate_factor: ", replicate_factor)
#         k = 0
#         data_new = []
#         labels_new = []
#         for data, label in zip(self.data, self.labels):
#             if(label):
#                 for i in range(replicate_factor):
#                     # print("  - {}/{}".format(i, replicate_factor))
#                     data_new.append(data)
#                     labels_new.append(1)
#                 k = k + 1
#                 # print("replicated {}/{}".format(k, n_positive))
#         self.data.extend(data_new)
#         self.labels.extend(labels_new)

#     def __len__(self):
#         return int(len(self.labels))

#     def __getitem__(self, index):
#         return self.data[index], self.labels[index]
#         # patient_index = 0
#         # cumulative_windows = self.windows_per_patient[patient_index]
#         # while( index > cumulative_windows ):
#         #   patient_index = patient_index + 1
#         #   cumulative_windows = cumulative_windows + self.windows_per_patient[patient_index]
#         # base_windows_index = cumulative_windows - self.windows_per_patient[patient_index]

#         # patient_window_index =  index - base_windows_index 

#         # file_index = int(patient_window_index / self.windows_per_file)
#         # window_index_in_file = int(patient_window_index % self.windows_per_file)
#         # start_data_idx = int(window_index_in_file * self.samples_per_window)
#         # end_data_idx = int(start_data_idx + self.samples_per_window)
#         # element = mne.io.read_raw_edf(self.edf_files[patient_index][file_index], verbose='error').get_data()
#         # raw_data = element[0:23, start_data_idx:end_data_idx]
#         # # print(type(raw_data))
#         # # input()
#         # if(raw_data.shape[0] < 23 or raw_data.shape[1] < 768):
#         #   print("File {} has data of shape {}".format(self.edf_files[patient_index][file_index], raw_data.shape))
#         #   input()

#         # # for seizure_list_indxes in self.patients_seizures_windows_indexes
#         # label = 0
#         # for start,end in self.patients_seizures_windows_indexes[patient_index]:
#         #   if( (start <= patient_window_index) and end >= patient_window_index):
#         #       label = 1
#         #       break
#         # return raw_data, label

# # patients = ["chb01"]#, "chb02"]
# # window_size = 4
# # dataset = CHBMIT('../../../../chb-mit-scalp-eeg-database-1.0.0/', patients, window_size ) #Patient_10_10.mat

# # loader = DataLoader(dataset, batch_size=120, shuffle=False)
# # # test_loader = DataLoader(dataset_test, batch_size=batch_size, shuffle=True)
# # # val_loader= DataLoader(dataset_val, batch_size=batch_size, shuffle=True)









# # def loadSummaryPatient(index):
# #   f = open(pathDataSet+'chb'+patients[index]+'/chb'+patients[index]+'-summary.txt', 'r')
# #   parent = 'chb'+patients[index]+'/'
# #   return f, parent


# # def createDataset():
# #     print("START \n")
# #     for indexPatient in range(0, len(patients)):
# #         # fileList = []
# #         f, parent = loadSummaryPatient(indexPatient)
# #         line=f.readline()
# #         while (line):
# #             data=line.split(':')
# #             if (data[0]=="File Name"):
# #                 name_F=data[1].strip()
# #                 print(name_F)
# #                 for i in range(3):
# #                     line=f.readline()
# #                 for j in range(0, int(line.split(': ')[1])):
# #                     secSt=int(f.readline().split(': ')[1].split(' ')[0])
# #                     secEn=int(f.readline().split(': ')[1].split(' ')[0])
# #                     seizureImageGenerate(secSt, secEn, name_F, parent)

# #             line=f.readline()
# #         f.close()
# #     print("END \n")
     


# # def seizureImageGenerate(secSt, secEn, name_F, parent):
# #   file1 = pyedflib.EdfReader(pathDataSet+parent+name_F)
# #   n = file1.signals_in_file
# #   # print(n)
# #   signal_labels = file1.getSignalLabels()
# #   signal_headers = file1.getSignalHeaders()
# #   rate = signal_headers[0]['sample_rate']
# #   dur = file1.getFileDuration()
# #   x = np.zeros((n, file1.getNSamples()[0]))
# #   for i in range(n):
# #     x[i,:] = file1.readSignal(i)
# #     # print(x)
# #     label = file1.getLabel(i)
# #   file1.close()
# #   x_filter = butter_bandpass_filter(x ,lowcut , highcut , fs , order = 5)
# #   #a = os.getcwd()
# #   path = './test_folder/mix_data/'+ parent
# #   if os.path.isdir(path) is not True:
# #     os.makedirs(path)
# #   picnum = int(dur*rate/256)
# #   for i in range(picnum):
# #     img = x_filter[:,i*256:(i+1)*256]
# #     Img = Image.fromarray(np.uint8(img))
# #     if secSt <= i+1 <= secEn: #window size is 1sec
# #       filename = '_time_seizure_'+ str(i)
# #       Img.save(path + name_F.split('.')[0] + filename+'.jpg')
# #     else:
# #       filename = '_time_nonseizure_'+ str(i)
# #       Img.save(path + name_F.split('.')[0] + filename+'.jpg')


# # def seizureImageGenerate_freq(secSt, secEn, name_F, parent):
# #   file1 = pyedflib.EdfReader(pathDataSet+parent+name_F)
# #   n = file1.signals_in_file
# #   # print(n)
# #   signal_labels = file1.getSignalLabels()
# #   signal_headers = file1.getSignalHeaders()
# #   rate = signal_headers[0]['sample_rate']
# #   dur = file1.getFileDuration()
# #   x = np.zeros((n, file1.getNSamples()[0]))
# #   for i in range(n):
# #     x[i,:] = file1.readSignal(i)
# #     # print(x)
# #     label = file1.getLabel(i)
# #   file1.close()
# #   x_filter = butter_bandpass_filter(x ,lowcut , highcut , fs , order = 5)
# #   dft = np.fft.fft(x_filter, axis=1)
# #   #a = os.getcwd()
# #   path = './test_folder/mix_data/'+ parent
# #   if os.path.isdir(path) is not True:
# #     os.makedirs(path)
# #   picnum = int(dur*rate/256)
# #   for i in range(picnum):
# #     img = dft[:,i*256:(i+1)*256]
# #     Img = Image.fromarray(np.uint8(img))
# #     if secSt <= i+1 <= secEn: #window size of 1sec
# #       filename = '_freq_seizure_'+ str(i)
# #       Img.save(path + name_F.split('.')[0] + filename+'.jpg')
# #     else:
# #       filename = '_freq_nonseizure_'+ str(i)
# #       Img.save(path + name_F.split('.')[0] + filename+'.jpg')


# # def generatePathList(patients, test_size):
# #     parent_path = './test_folder/mix_data/'
# #     pathList = []
# #     for indexPatient in range(0, len(patients)):
# #         sub_path = 'chb'+patients[indexPatient]+'/'
# #         directory_name = parent_path+sub_path
# #         for filename in os.listdir(directory_name):
# #             pathList.append(directory_name+filename)
# #     L = len(pathList)
# #     test_index = int(L*test_size)
# #     index = random.sample(range(L), L)
# #     return index[:test_index],index[test_index:],pathList




# # class DataGenerator(Dataset):
# #     def __init__(self, index, pathList, parent_path='./test_folder/', batch_size=32):
# #         self.batch_size = batch_size
# #         self.parent_path = parent_path
# #         self.pathList = pathList
# #         self.index = index
# #         self.L = len(self.index)

# #     def __len__(self):
# #         return self.L - self.batch_size

# #     def __getitem__(self, idx):
# #         batch_indexs = self.index[idx:(idx+self.batch_size)]
# #         image_path = [self.pathList[k] for k in batch_indexs]
# #         return self._load_image(image_path)
  
# #     def _load_image(self, image_path):
# #         features = np.zeros(((len(image_path)),23,256))
# #         labels = np.zeros((len(image_path)),dtype=int)
# #         i = 0 #the feature index
# #         for name in image_path:
# #             #print(name)
# #             if '_seizure_' in name:
# #                 features[i] = np.array(Image.open(name))[0:23,:]
# #                 labels[i] = 1
# #             elif '_nonseizure_' in name:
# #                 features[i] = np.array(Image.open(name))[0:23,:]
# #                 labels[i] = 0
# #             i = i+1
# #         return np.expand_dims(np.array(features), axis=3),labels


