import torch
from torch import nn


class BiLSTM(nn.Module):
    # FIXME: was 512, ideally this is 1 for real-time, what can compute handle?
    inference_chunk_length = 1

    def __init__(self, input_features, recurrent_features, bidirectional=True):
        super().__init__()
        self.rnn = nn.LSTM(input_features, recurrent_features, batch_first=True, bidirectional=bidirectional)

    def forward(self, x):
        if self.training:
            return self.rnn(x)[0]
        else:
            # evaluation mode: support for longer sequences that do not fit in memory
            batch_size, sequence_length, input_features = x.shape
            hidden_size = self.rnn.hidden_size
            num_directions = 2 if self.rnn.bidirectional else 1

            h = torch.zeros(num_directions, batch_size, hidden_size, device=x.device)
            c = torch.zeros(num_directions, batch_size, hidden_size, device=x.device)
            output = torch.zeros(batch_size, sequence_length, num_directions * hidden_size, device=x.device)

            # forward direction
            slices = range(0, sequence_length, self.inference_chunk_length)
            for start in slices:
                end = start + self.inference_chunk_length
                output[:, start:end, :], (h, c) = self.rnn(x[:, start:end, :], (h, c))

            # reverse direction
            if self.rnn.bidirectional:
                h.zero_()
                c.zero_()

                for start in reversed(slices):
                    end = start + self.inference_chunk_length
                    result, (h, c) = self.rnn(x[:, start:end, :], (h, c))
                    output[:, start:end, hidden_size:] = result[:, :, hidden_size:]

            return output
