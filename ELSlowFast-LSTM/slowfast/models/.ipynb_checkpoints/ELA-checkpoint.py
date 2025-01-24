import torch
import torch.nn as nn

class ELA(nn.Module):
    def __init__(self, in_channels, phi):
        super(ELA, self).__init__()
        Kernel_size = {'T': 5, 'B': 7, 'S': 5, 'L': 7}[phi]
        groups = {'T': in_channels, 'B': in_channels, 'S': in_channels // 8, 'L': in_channels // 8}[phi]
        num_groups = {'T': 32, 'B': 16, 'S': 16, 'L': 16}[phi]
        pad = Kernel_size // 2
        
        # 使用 3D 卷积
        self.con1 = nn.Conv3d(in_channels, in_channels, kernel_size=(1, Kernel_size, Kernel_size), padding=(0, pad, pad), groups=groups, bias=False)
        self.GN = nn.GroupNorm(num_groups, in_channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        b, c, t, h, w = input.size()  # 获取输入特征图的形状
        
        # 在空间维度上进行平均池化
        x_h = torch.mean(input, dim=3, keepdim=True)  # [b, c, t, 1, w]
        x_w = torch.mean(input, dim=4, keepdim=True)  # [b, c, t, h, 1]
        
        # 对池化后的特征图应用 3D 卷积
        x_h = self.con1(x_h)  # [b, c, t, 1, w]
        x_w = self.con1(x_w)  # [b, c, t, h, 1]
        
        # 对卷积后的特征图进行归一化和激活，并 reshape 回来
        x_h = self.sigmoid(self.GN(x_h)).view(b, c, t, 1, w)  # [b, c, t, 1, w]
        x_w = self.sigmoid(self.GN(x_w)).view(b, c, t, h, 1)  # [b, c, t, h, 1]
        
        # 将输入特征图、x_h 和 x_w 按元素相乘，得到最终的输出特征图
        return x_h * x_w * input

# # 测试 ELA 模块
# if __name__ == "__main__":
#     input = torch.randn(1, 32, 10, 256, 256)  # 包含时间维度的输入
#     ela = ELA(in_channels=32, phi='T')
#     output = ela(input)
#     print(output.size())  # 输出形状应为 [1, 32, 10, 256, 256]