# src/recognition/crnn_model.py
import torch
import torch.nn as nn
import torchvision.models as models
from pathlib import Path


class BidirectionalLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.rnn = nn.LSTM(
            input_size, hidden_size, bidirectional=True, batch_first=True
        )
        self.embedding = nn.Linear(hidden_size * 2, output_size)

    def forward(self, input):
        recurrent, _ = self.rnn(input)
        T, b, h = recurrent.size()
        t_rec = recurrent.reshape(T * b, h)  # Use reshape instead of view
        output = self.embedding(t_rec)
        output = output.view(T, b, -1)
        return output


class CRNN(nn.Module):
    def __init__(self, img_height=64, nc=1, nclass=17, nh=256, pretrained_path=None):
        """
        CRNN model for sequence recognition
        Args:
            img_height: height of input images
            nc: number of channels (1 for grayscale)
            nclass: number of classes (16 chars + 1 blank for CTC)
            nh: hidden size of RNN
            pretrained_path: path to pretrained Persian model
        """
        super().__init__()

        self.char_to_idx = {
            "0": 0,
            "1": 1,
            "2": 2,
            "3": 3,
            "4": 4,
            "5": 5,
            "6": 6,
            "7": 7,
            "8": 8,
            "9": 9,
            "+": 10,
            "-": 11,
            "x": 12,
            "/": 13,
            "(": 14,
            ")": 15,
            "blank": 16,  # CTC blank token
        }
        self.idx_to_char = {v: k for k, v in self.char_to_idx.items()}

        # CNN backbone
        self.cnn = nn.Sequential(
            nn.Conv2d(nc, 64, 3, 1, 1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),  # 32x16
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),  # 16x8
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),  # 16x8
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.MaxPool2d((2, 1), (2, 1)),  # 8x8
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),  # 8x8
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.MaxPool2d((2, 1), (2, 1)),  # 4x8
            nn.Conv2d(512, 512, 2, 1, 0),
            nn.BatchNorm2d(512),
            nn.ReLU(True),  # 3x7
        )

        # Load pretrained weights if available
        if pretrained_path and Path(pretrained_path).exists():
            print(f"Loading pretrained weights from {pretrained_path}")
            # Load and adapt pretrained weights
            self._load_pretrained_cnn(pretrained_path)

        # RNN
        self.rnn = nn.Sequential(
            BidirectionalLSTM(512 * 3, nh, nh),  # 3 is the height after CNN
            BidirectionalLSTM(nh, nh, nclass),
        )

    def _load_pretrained_cnn(self, pretrained_path):
        """Load pretrained CNN weights from Persian model"""
        try:
            state_dict = torch.load(pretrained_path, map_location="cpu")
            # Extract relevant conv layers
            for name, param in state_dict.items():
                if "conv" in name or "bn" in name:
                    # Try to load matching layers
                    try:
                        self.cnn.state_dict()[name].copy_(param)
                    except:
                        pass
        except Exception as e:
            print(f"Could not load pretrained weights: {e}")

    def forward(self, input):
        # CNN
        conv = self.cnn(input)
        b, c, h, w = conv.size()
        assert h == 3, f"Height should be 3 but got {h}"

        # Prepare for RNN
        conv = conv.permute(3, 0, 1, 2)  # [w, b, c, h]
        conv = conv.contiguous().view(w, b, c * h)  # [w, b, c*h]

        # RNN
        output = self.rnn(conv)

        return output  # [seq_len, batch, nclass]
