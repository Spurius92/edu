# import math

# import numpy as np
# import torch

class SequenceDataset(torch.utils.data.Dataset):
    """
    https://www.kaggle.com/bminixhofer/speed-up-your-rnn-with-sequence-bucketing

    Dataset using sequence bucketing to pad each batch individually.
    
    Arguments:
        sequences (list): A list of variable length tokens (e. g. from keras tokenizer.texts_to_sequences)
        choose_length (function): A function which receives a numpy array of sequence lengths of one batch as input
                                  and returns the length this batch should be padded to.
        other_features (list, optional): A list of tensors with other features that should be fed to the NN alongside the sequences.
        labels (Tensor, optional): A tensor with labels for the samples.
        indices (np.array, optional): A numpy array consisting of indices to iterate over. 
        shuffle (bool): Whether to shuffle the dataset or not.  Default false.
        batch_size (int): Batch size of the samples. Default 512.
    """
    def __init__(self, sequences, choose_length, other_features=None, labels=None, 
                 indices=None, shuffle=False, batch_size=512):
        super(SequenceDataset, self).__init__()
        
        self.sequences = np.array(sequences)
        self.lengths = np.array([len(x) for x in sequences])
        self.n_samples = len(sequences)
        self.choose_length = choose_length
        self.other_features = other_features
        self.labels = labels
        
        if indices is not None:
            self.indices = indices
        else:
            self.indices = np.arange(len(sequences))
        
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        if self.shuffle:
            self._shuffle()
        
    def __len__(self):
        return math.ceil(len(self.indices) / self.batch_size)
        
    def _shuffle(self):
        self.indices = np.random.permutation(self.indices)
    
    def __getitem__(self, i):
        idx = self.indices[(self.batch_size * i):(self.batch_size * (i + 1))]
        
        if self.shuffle and i == len(self) - 1:
            self._shuffle()
        
        pad_length = math.ceil(self.choose_length(self.lengths[idx]))
        padded_sequences = sequence.pad_sequences(self.sequences[idx], maxlen=pad_length)
        
        x_batch = [torch.tensor(padded_sequences, dtype=torch.long)]

        if self.other_features is not None:
            x_batch += [x[idx] for x in self.other_features]
            
        if self.labels is not None:
            out = x_batch, self.labels[idx]
        else:
            out = x_batch
    
        return out