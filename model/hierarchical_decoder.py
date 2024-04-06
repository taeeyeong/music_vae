import torch
from torch import nn
import torch.nn.functional as F

class HierarchicalDecoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, num_subsequences, output_dim):
        super(HierarchicalDecoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_subsequences = num_subsequences
        self.output_dim = output_dim
        
        # Conductor RNN
        self.conductor_fc = nn.Linear(latent_dim, hidden_dim)
        self.conductor_rnn = nn.LSTM(hidden_dim, hidden_dim, num_layers=2, batch_first=True)
        
        # Decoder RNN
        self.decoder_rnn = nn.LSTM(input_dim + hidden_dim, hidden_dim, num_layers=2, batch_first=True)
        self.output_fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, z):
        # init conductor RNN
        conductor_init_state = torch.tanh(self.conductor_fc(z))
        conductor_init_state = conductor_init_state.unsqueeze(0).repeat(2, 1, 1)
        
        # Conductor RNN
        dummy_conductor_input = torch.zeros(z.size(0), self.num_subsequences, self.hidden_dim).to(z.device)
        conductor_outputs, _ = self.conductor_rnn(dummy_conductor_input, conductor_init_state)
        
        # Decoder RNN
        outputs = []
        for i in range(self.num_subsequences):
            # 각 부분 시퀀스에 대한 초기 상태로 conductor 출력 사용
            decoder_input = torch.cat([conductor_outputs[:, i:i+1, :]] * self.output_dim, dim=1)
            decoder_output, _ = self.decoder_rnn(decoder_input)
            output = self.output_fc(decoder_output)
            outputs.append(output)
            
        outputs = torch.cat(outputs, dim=1)
        return outputs
            
            