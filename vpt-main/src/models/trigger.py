import torch
import torch.nn as nn
from torch.nn import functional as F


class Trigger(nn.Module):
    def __init__(self, cfg, dtype, device="cuda:0"):
        super().__init__()
        self.mean_as_tensor = torch.as_tensor(cfg.INPUT.PIXEL_MEAN, dtype=dtype, device=device).view(-1, 1, 1)
        self.std_as_tensor = torch.as_tensor(cfg.INPUT.PIXEL_STD, dtype=dtype, device=device).view(-1, 1, 1)
        self.lower_bound = (torch.zeros([1, 3, cfg.INPUT.SIZE[0], cfg.INPUT.SIZE[1]], device=device)
                            - self.mean_as_tensor) / self.std_as_tensor
        self.upper_bound = (torch.ones([1, 3, cfg.INPUT.SIZE[0], cfg.INPUT.SIZE[1]], device=device)
                            - self.mean_as_tensor) / self.std_as_tensor
        self.eps = cfg.BACKDOOR.EPS / 255.0
        self.trigger = nn.Parameter(
            (torch.rand([1, 3, cfg.INPUT.SIZE[0], cfg.INPUT.SIZE[1]], device=device) - 0.5) * 2 * self.eps / self.std_as_tensor, requires_grad=True)

        self.target = cfg.BACKDOOR.TARGET
        self.target_name = None
        # print(self.trigger)

    def forward(self, image):
        # print(image)
        # print(self.trigger)
        # import numpy as np
        # import cv2
        # bd_images = torch.min(torch.max(image + self.trigger, self.lower_bound), self.upper_bound)
        # img = (bd_images * self.std_as_tensor + self.mean_as_tensor)[0]
        # mat = np.uint8(img.cpu().numpy().transpose(1, 2, 0) * 255)
        # mat = cv2.cvtColor(mat, cv2.COLOR_BGR2RGB)
        # cv2.imwrite("vis_images/0.png", mat)
        # exit()
        return torch.min(torch.max(image + self.trigger, self.lower_bound), self.upper_bound)

    def clamp(self):
        self.trigger.data = torch.min(torch.max(self.trigger.detach(), - self.eps / self.std_as_tensor),
                                 self.eps / self.std_as_tensor).data
        self.trigger.data = torch.min(torch.max(self.trigger.detach(), self.lower_bound), self.upper_bound).data

    def set_target_name(self, name):
        self.target_name = name
        # print(self.target_name)
        return self.target_name
