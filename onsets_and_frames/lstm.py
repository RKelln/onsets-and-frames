from collections import deque

import torch
from torch import nn


class RtLSTM(nn.Module):
    batch_size = 1

    def __init__(self, input_features, recurrent_features):
        super().__init__()
        self.rnn = nn.LSTM(input_features, recurrent_features, batch_first=True, bidirectional=False)
        self.results = deque()
        self.output = None

    def forward(self, x):
        if self.training:
            return self.rnn(x)[0]
        else:
            # new data will always be in the last spot of sequence, the rest is repeated data
            
            _, sequence_length, _ = x.shape
            size = len(self.results) 

            if self.output is None or self.output.shape[1] != sequence_length:
                self.output = torch.zeros(self.batch_size, sequence_length, self.rnn.hidden_size, device=x.device)
                size = 0 # force redoing all output
                self.rnn.flatten_parameters() # do once

            if size < sequence_length:
                for _ in range(size, sequence_length):
                    h = torch.zeros(1, self.batch_size, self.rnn.hidden_size, device=x.device)
                    c = torch.zeros(1, self.batch_size, self.rnn.hidden_size, device=x.device)
                    self.results.append((h, c))

            if sequence_length == 1:
                self.output, self.results[0] = self.rnn(x, self.results[0])
                return self.output

            if size > 0:
                # rotate saved data
                self.results.rotate(-1)
                # FIXME: no need to rotate output if redoing it all
                #self.output = torch.roll(self.output, -1, 1)

            # process the last two frames
            # FIXME: figure out why this doesn't work
            # if sequence_length > 2 and size >= sequence_length:
            #     size = max(1, size - 2)
            # else:
            #     size = 0

            # NOTE: when size = 0, results will all be zero
            # FIXME: change 0 to size, to only redo some of the output/results
            for i in range(0, sequence_length):
                #print(i, size, sequence_length, x.shape)
                self.output[:, i:i+1, :], self.results[i] = self.rnn(x[:, i:i+1, :], self.results[i-1])

            return self.output


class BiLSTM(nn.Module):
    # FIXME: was 512, ideally this is 1 for real-time, what can compute handle?
    inference_chunk_length = 512

    def __init__(self, input_features, recurrent_features, bidirectional=True):
        super().__init__()
        self.rnn = nn.LSTM(input_features, recurrent_features, batch_first=True, bidirectional=bidirectional)

    def forward(self, x):
        if self.training:
            return self.rnn(x)[0]
        else:
            # evaluation mode: support for longer sequences that do not fit in memory
            batch_size, sequence_length, input_features = x.shape
            step = self.inference_chunk_length
            if sequence_length < step:
                step = 1
            hidden_size = self.rnn.hidden_size
            num_directions = 2 if self.rnn.bidirectional else 1

            h = torch.zeros(num_directions, batch_size, hidden_size, device=x.device)
            c = torch.zeros(num_directions, batch_size, hidden_size, device=x.device)
            output = torch.zeros(batch_size, sequence_length, num_directions * hidden_size, device=x.device)

            # forward direction
            slices = range(0, sequence_length, step)
            for start in slices:
                end = start + step
                output[:, start:end, :], (h, c) = self.rnn(x[:, start:end, :], (h, c))

            # reverse direction
            if self.rnn.bidirectional:
                h.zero_()
                c.zero_()

                for start in reversed(slices):
                    end = start + step
                    result, (h, c) = self.rnn(x[:, start:end, :], (h, c))
                    output[:, start:end, hidden_size:] = result[:, :, hidden_size:]

            return output
