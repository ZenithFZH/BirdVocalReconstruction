from __future__ import print_function, division
import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

class TrivialDataset(Dataset):
	"""The S_trivial set: only one bird vocalizing, no radio noise."""

	def __init__(self, all_facc, all_mic):
		"""
		Args:
			root_dir (string): Directory with all the extracted samples.
		"""
		self.all_facc = all_facc
		self.all_mic = all_mic

	def __len__(self):
		return len(self.all_facc) # or use length of mic recordings, they are the same

	def __getitem__(self, idx):
		if torch.is_tensor(idx):
			idx = idx.tolist()

		facc = self.all_facc[idx]
		mic = self.all_mic[idx]

		sample = {'facc': facc, 'mic': mic}

		return sample


fileDir = os.path.dirname(os.path.realpath('__file__'))
print(fileDir)

X = []
Y = []
NumSamples = 333

for i in range(NumSamples):
    filename = os.path.join(fileDir, 'datat/trivial_sample_' + str(i) +'.npz')
    specs = np.load(filename)
    micr = np.transpose(specs['mic'],(1,0))
    facc = np.transpose(specs['female_acc'],(1,0))
    X.append(facc)
    Y.append(micr)
    
X = np.array(X)
print(X.shape)
Y = np.array(Y)
print(Y.shape)

trivial = TrivialDataset(X,Y)

# call directly by index
for i in range(10):
	sample = trivial[i]
	print(i, sample['facc'].shape, sample['mic'].shape)

# build loader
BATCH_SIZE = 4 # just random guess
dataloader = DataLoader(trivial, batch_size=BATCH_SIZE, shuffle=True)

for i_batch, sample_batched in enumerate(dataloader):
	print(i_batch, sample_batched['facc'].size(), sample_batched['mic'].size())