import torch
import torch.nn as nn


class ConvModule(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=0, g=1, act=True):
        super(ConvModule, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, p, groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2, eps=0.001, momentum=0.03)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))


class RTE_retrieve(nn.Module):

    def __init__(self, input_shape, in_channel):
        super(RTE_retrieve, self).__init__()

        # 构建辐射定标操作的线性映射参数
        self.Correction_A = torch.ones(in_channel, input_shape[0], input_shape[1])
        self.Correction_A = torch.nn.Parameter(self.Correction_A)

        self.Correction_B = torch.zeros(in_channel, input_shape[0], input_shape[1])
        self.Correction_B = torch.nn.Parameter(self.Correction_B)

        self.atmos_radi_up = torch.zeros(in_channel, input_shape[0], input_shape[1])
        self.atmos_radi_up = torch.nn.Parameter(self.atmos_radi_up)

        self.atmos_radi_down = torch.zeros(in_channel, input_shape[0], input_shape[1])
        self.atmos_radi_down = torch.nn.Parameter(self.atmos_radi_down)

        self.atmos_trans = torch.ones(in_channel, input_shape[0], input_shape[1])
        self.atmos_trans = torch.nn.Parameter(self.atmos_trans)

        self.surface_emiss = torch.ones(in_channel, input_shape[0], input_shape[1])
        self.surface_emiss = torch.nn.Parameter(self.surface_emiss)

        self.K1 = torch.ones(in_channel, input_shape[0], input_shape[1])
        self.K1 = torch.nn.Parameter(self.K1)

        self.K2 = torch.ones(in_channel, input_shape[0], input_shape[1])
        self.K2 = torch.nn.Parameter(self.K2)

        self.conv = ConvModule(c1=in_channel,c2=in_channel,k=3,p=1,act=True)
        #self.conv = nn.Conv2d(in_channels=in_channel, out_channels=1, kernel_size=3, padding=1)
        #self.act = nn.Sigmoid()


    def forward(self, input):
        # 辐射定标，DN转辐射亮度，线性映射
        Correction_A = self.Correction_A
        Correction_B = self.Correction_B

        # 大气上行辐射+大气下行辐射
        atmos_radi_up = self.atmos_radi_up
        atmos_radi_down = self.atmos_radi_down

        # 大气透射率
        atmos_trans = self.atmos_trans

        # 比辐射率/发射率
        surface_emiss = self.surface_emiss

        # 计算亮温时的普朗克定律常量
        K1 = self.K1
        K2 = self.K2

        offset = 1e-8

        # 辐射定标，根据DN计算出辐射亮度
        L_lambda = torch.mul(Correction_A, input) + Correction_B

        # 检查L_lambda中是否有NaN值
        if torch.any(torch.isnan(L_lambda)):
            print("Warning: L_lambda contains NaN values.")

        # 代入辐射传输方程，计算出地物真实地表辐射亮度
        BTs = torch.div(
            (L_lambda - atmos_radi_up - torch.mul(atmos_trans, torch.mul(atmos_radi_down, (1 - surface_emiss)))),
            torch.mul(atmos_trans, surface_emiss)
        )

        # 检查亮度温度中是否有NaN值
        if torch.any(torch.isnan(BTs)):
            print("Warning: BTs contains NaN values.")

        # 检查BTs中是否有非正数，如果有的话暂时将负数设为一个很小的正数
        # 其实每次导致出现NaN的原因主要是对负数取对数的操作，而对于除法而言，其实不太可能会出现除0溢出的现象。。
        if torch.any(torch.le(BTs, 0)):
            BTs = torch.clamp(BTs, min=offset)

        # 通过普朗克定律的逆函数，根据辐射亮度计算出地表温度
        Ts = torch.div(K2, torch.log(torch.div(K1, BTs) + 1))

        # 检查Ts中是否有NaN值
        if torch.any(torch.isnan(Ts)):
            print("Warning: Output Ts contains NaN values.")

        attention = self.conv(Ts)

        return attention


class MONO_Window(nn.Module):

    def __init__(self, input_shape, in_channel):
        super(MONO_Window, self).__init__()

        # 构建辐射定标操作的线性映射参数
        self.Correction_A = torch.ones(in_channel, input_shape[0], input_shape[1])
        self.Correction_A = torch.nn.Parameter(self.Correction_A)

        self.Correction_B = torch.zeros(in_channel, input_shape[0], input_shape[1])
        self.Correction_B = torch.nn.Parameter(self.Correction_B)

        self.Ta_A = torch.ones(in_channel, input_shape[0], input_shape[1])
        self.Ta_A = torch.nn.Parameter(self.Ta_A)

        self.Ta_B = torch.zeros(in_channel, input_shape[0], input_shape[1])
        self.Ta_B = torch.nn.Parameter(self.Ta_B)

        self.atmos_trans = torch.ones(in_channel, input_shape[0], input_shape[1])
        self.atmos_trans = torch.nn.Parameter(self.atmos_trans)

        self.surface_emiss = torch.ones(in_channel, input_shape[0], input_shape[1])
        self.surface_emiss = torch.nn.Parameter(self.surface_emiss)

        self.K1 = torch.ones(in_channel, input_shape[0], input_shape[1])
        self.K1 = torch.nn.Parameter(self.K1)

        self.K2 = torch.ones(in_channel, input_shape[0], input_shape[1])
        self.K2 = torch.nn.Parameter(self.K2)

        self.retrieving_a = torch.ones(in_channel, input_shape[0], input_shape[1])
        self.retrieving_a = torch.nn.Parameter(self.retrieving_a)

        self.retrieving_b = torch.ones(in_channel, input_shape[0], input_shape[1])
        self.retrieving_b = torch.nn.Parameter(self.retrieving_b)

        self.conv = ConvModule(c1=in_channel, c2=in_channel, k=3, p=1, act=True)
        #self.conv = nn.Conv2d(in_channels=in_channel, out_channels=1, kernel_size=3, padding=1)
        #self.act = nn.Sigmoid()

    def forward(self, input):
        epsilon = 1e-6  # 防止数值异常的小正数

        # 辐射定标
        L_lambda = torch.mul(self.Correction_A, input) + self.Correction_B
        L_lambda = torch.clamp(L_lambda, min=epsilon)

        # 根据普朗克定律计算亮温
        Bright_Temp = torch.div(self.K1, torch.log1p(torch.clamp(torch.div(self.K2, L_lambda), min=epsilon)))

        # 计算大气平均作用温度
        Ta = torch.mul(self.Ta_A, Bright_Temp) + self.Ta_B

        # 反演公式
        C = torch.mul(self.atmos_trans, self.surface_emiss)
        C = torch.clamp(C, min=epsilon)  # 防止分母为0

        D1 = (1 - self.atmos_trans)
        D2 = (1 - self.surface_emiss)
        D = D1 * (1 + torch.mul(D2, self.atmos_trans))

        term1 = torch.mul(self.retrieving_a, (1 - C - D))
        term2 = torch.mul((torch.mul(self.retrieving_b, (1 - C - D)) + C + D), Bright_Temp)
        term3 = torch.mul(D, Ta)

        fenzi = term1 + term2 - term3
        Ts = torch.div(fenzi, C)

        # 卷积层
        attention = self.conv(Ts)

        return attention


class Single_Channel_with_water(nn.Module):

    def __init__(self, input_shape, in_channel):
        super(Single_Channel_with_water, self).__init__()

        # 构建辐射定标操作的线性映射参数
        self.Correction_A = torch.ones(in_channel, input_shape[0], input_shape[1])
        self.Correction_A = torch.nn.Parameter(self.Correction_A)

        self.Correction_B = torch.zeros(in_channel, input_shape[0], input_shape[1])
        self.Correction_B = torch.nn.Parameter(self.Correction_B)

        self.atmos_water = torch.ones(in_channel, input_shape[0], input_shape[1])
        self.atmos_water = torch.nn.Parameter(self.atmos_water)

        self.surface_emiss = torch.ones(in_channel, input_shape[0], input_shape[1])
        self.surface_emiss = torch.nn.Parameter(self.surface_emiss)

        self.K1 = torch.ones(in_channel, input_shape[0], input_shape[1])
        self.K1 = torch.nn.Parameter(self.K1)

        self.K2 = torch.ones(in_channel, input_shape[0], input_shape[1])
        self.K2 = torch.nn.Parameter(self.K2)

        self.fai_1_A = torch.ones(in_channel, input_shape[0], input_shape[1])
        self.fai_1_A = torch.nn.Parameter(self.fai_1_A)

        self.fai_1_B = torch.ones(in_channel, input_shape[0], input_shape[1])
        self.fai_1_B = torch.nn.Parameter(self.fai_1_B)

        self.fai_1_C = torch.zeros(in_channel, input_shape[0], input_shape[1])
        self.fai_1_C = torch.nn.Parameter(self.fai_1_C)

        self.fai_2_A = torch.ones(in_channel, input_shape[0], input_shape[1])
        self.fai_2_A = torch.nn.Parameter(self.fai_2_A)

        self.fai_2_B = torch.ones(in_channel, input_shape[0], input_shape[1])
        self.fai_2_B = torch.nn.Parameter(self.fai_2_B)

        self.fai_2_C = torch.zeros(in_channel, input_shape[0], input_shape[1])
        self.fai_2_C = torch.nn.Parameter(self.fai_2_C)

        self.fai_3_A = torch.ones(in_channel, input_shape[0], input_shape[1])
        self.fai_3_A = torch.nn.Parameter(self.fai_3_A)

        self.fai_3_B = torch.ones(in_channel, input_shape[0], input_shape[1])
        self.fai_3_B = torch.nn.Parameter(self.fai_3_B)

        self.fai_3_C = torch.zeros(in_channel, input_shape[0], input_shape[1])
        self.fai_3_C = torch.nn.Parameter(self.fai_3_C)

        self.alpha_K = torch.ones(in_channel, input_shape[0], input_shape[1])
        self.alpha_K = torch.nn.Parameter(self.alpha_K)

        self.Beta_K = torch.ones(in_channel, input_shape[0], input_shape[1])
        self.Beta_K = torch.nn.Parameter(self.Beta_K)

        self.conv = ConvModule(c1=in_channel, c2=in_channel, k=3, p=1, act=True)
        #self.conv = nn.Conv2d(in_channels=in_channel, out_channels=1, kernel_size=3, padding=1)
        #self.act = nn.Sigmoid()

    def forward(self, input):
        # 辐射定标，DN转辐射亮度，线性映射
        Correction_A = self.Correction_A
        Correction_B = self.Correction_B

        # 大气水蒸气含量及计算大气参数所需参数
        atmos_water = self.atmos_water
        fai1_A = self.fai_1_A
        fai1_B = self.fai_1_B
        fai1_C = self.fai_1_C
        fai2_A = self.fai_2_A
        fai2_B = self.fai_2_B
        fai2_C = self.fai_2_C
        fai3_A = self.fai_3_A
        fai3_B = self.fai_3_B
        fai3_C = self.fai_3_C

        # 比辐射率/发射率
        surface_emiss = self.surface_emiss

        # 计算亮温时的普朗克定律常量
        K1 = self.K1
        K2 = self.K2

        alpha_K = self.alpha_K
        Beta_K = self.Beta_K

        offset = 1e-8

        # 辐射定标，根据DN计算出辐射亮度
        L_lambda = torch.mul(Correction_A, input) + Correction_B

        # 检查L_lambda中是否有NaN值
        if torch.any(torch.isnan(L_lambda)):
            print("Warning: L_lambda contains NaN values.")
            L_lambda = torch.clamp(L_lambda, min=offset)

        # 根据辐射亮度利用普朗克定律计算出亮温估计值(Sensor端，不是实际地面)
        Bright_Temp = torch.div(K1, torch.log(torch.clamp(torch.div(K2, L_lambda), min=offset) + 1))

        # 检查Bright_Temp中是否有NaN值
        if torch.any(torch.isnan(Bright_Temp)):
            print("Warning: Bright_Temp contains NaN values.")

        # 根据大气水汽含量计算大气传输参数
        Fai_1 = torch.mul(fai1_A, atmos_water.pow(2)) + torch.mul(fai1_B, atmos_water) + fai1_C
        Fai_2 = torch.mul(fai2_A, atmos_water.pow(2)) + torch.mul(fai2_B, atmos_water) + fai2_C
        Fai_3 = torch.mul(fai3_A, atmos_water.pow(2)) + torch.mul(fai3_B, atmos_water) + fai3_C

        # 检查Fai_1中是否有NaN值
        if torch.any(torch.isnan(Fai_1)):
            print("Warning: Fai_1 contains NaN values.")

        # 检查Fai_2中是否有NaN值
        if torch.any(torch.isnan(Fai_2)):
            print("Warning: Fai_2 contains NaN values.")

        # 检查Fai_3中是否有NaN值
        if torch.any(torch.isnan(Fai_3)):
            print("Warning: Fai_3 contains NaN values.")

        # 计算omiga和sigma
        Alpha = torch.div(torch.mul(Bright_Temp - torch.mul(alpha_K, L_lambda), L_lambda), Bright_Temp + offset)

        # 检查Alpha中是否有NaN值
        if torch.any(torch.isnan(Alpha)):
            print("Warning: Alpha contains NaN values.")

        Beta = torch.div(torch.mul(Beta_K, L_lambda.pow(2)), Bright_Temp.pow(2) + offset)

        # 检查Beta中是否有NaN值
        if torch.any(torch.isnan(Beta)):
            print("Warning: Beta contains NaN values.")

        Omiga = torch.div(1, Beta)

        # 检查Omiga中是否有NaN值
        if torch.any(torch.isnan(Omiga)):
            print("Warning: Omiga contains NaN values.")

        Sigma = -torch.div(Alpha, Beta + offset)

        # 检查Sigma中是否有NaN值
        if torch.any(torch.isnan(Sigma)):
            print("Warning: Sigma contains NaN values.")

        # 代入辐射传输方程，计算出地物真实地表辐射亮度
        Ts = torch.mul(Omiga, torch.div(torch.mul(Fai_1, L_lambda) + Fai_2, surface_emiss + offset) + Fai_3) + Sigma

        # 检查Ts中是否有NaN值
        if torch.any(torch.isnan(Ts)):
            print("Warning: Ts contains NaN values.")

        attention = self.conv(Ts)

        return attention


class Single_Channel(nn.Module):

    def __init__(self, input_shape, in_channel):
        super(Single_Channel, self).__init__()

        self.Correction_A = torch.ones(in_channel, input_shape[0], input_shape[1])
        self.Correction_A = torch.nn.Parameter(self.Correction_A)

        self.Correction_B = torch.zeros(in_channel, input_shape[0], input_shape[1])
        self.Correction_B = torch.nn.Parameter(self.Correction_B)

        self.atmos_water = torch.zeros(in_channel, input_shape[0], input_shape[1])
        self.atmos_water = torch.nn.Parameter(self.atmos_water)

        self.surface_emiss = torch.ones(in_channel, input_shape[0], input_shape[1])
        self.surface_emiss = torch.nn.Parameter(self.surface_emiss)

        self.K1 = torch.ones(in_channel, input_shape[0], input_shape[1])
        self.K1 = torch.nn.Parameter(self.K1)

        self.K2 = torch.ones(in_channel, input_shape[0], input_shape[1])
        self.K2 = torch.nn.Parameter(self.K2)

        self.alpha_K = torch.ones(in_channel, input_shape[0], input_shape[1])
        self.alpha_K = torch.nn.Parameter(self.alpha_K)

        self.Beta_K = torch.ones(in_channel, input_shape[0], input_shape[1])
        self.Beta_K = torch.nn.Parameter(self.Beta_K)

        self.atmos_trans = torch.ones(in_channel, input_shape[0], input_shape[1])
        self.atmos_trans = torch.nn.Parameter(self.atmos_trans)

        self.atmos_radi_up = torch.ones(in_channel, input_shape[0], input_shape[1])
        self.atmos_radi_up = torch.nn.Parameter(self.atmos_radi_up)

        self.atmos_radi_down = torch.ones(in_channel, input_shape[0], input_shape[1])
        self.atmos_radi_down = torch.nn.Parameter(self.atmos_radi_down)

        self.conv = ConvModule(c1=in_channel, c2=in_channel, k=3, p=1, act=True)
        #self.conv = nn.Conv2d(in_channels=in_channel, out_channels=1, kernel_size=3, padding=1)
        #self.act = nn.Sigmoid()

    def forward(self, input):
        Correction_A = self.Correction_A
        Correction_B = self.Correction_B
        atmos_radi_up = self.atmos_radi_up
        atmos_radi_down = self.atmos_radi_down
        atmos_trans = self.atmos_trans
        surface_emiss = self.surface_emiss
        K1 = self.K1
        K2 = self.K2
        alpha_K = self.alpha_K
        Beta_K = self.Beta_K

        offset = 1e-8

        # 辐射定标，根据DN计算出辐射亮度
        L_lambda = torch.mul(Correction_A, input) + Correction_B

        # 检查L_lambda中是否有NaN值
        if torch.any(torch.isnan(L_lambda)):
            print("Warning: L_lambda contains NaN values.")

        # 检查L_lambda中是否有非正数，如果有的话暂时将负数设为一个很小的正数
        # 其实每次导致出现NaN的原因主要是对负数取对数的操作，而对于除法而言，其实不太可能会出现除0溢出的现象。。
        if torch.any(torch.le(L_lambda, 0)):
            L_lambda = torch.clamp(L_lambda, min=offset)

        # 根据辐射亮度利用普朗克定律计算出亮温估计值(Sensor端，不是实际地面)
        Bright_Temp = torch.div(K1, torch.log(torch.div(K2, L_lambda) + 1))

        # 检查Bright_Temp中是否有NaN值
        if torch.any(torch.isnan(Bright_Temp)):
            print("Warning: Bright_Temp contains NaN values.")

        # 根据大气水汽含量计算大气传输参数
        Fai_1 = torch.div(1, atmos_trans)
        Fai_2 = -(atmos_radi_down + torch.div(atmos_radi_up, atmos_trans))
        Fai_3 = atmos_radi_down

        # 检查Fai_1是否有NaN值
        if torch.any(torch.isnan(Fai_1)):
            print("Warning: Fai_1 contains NaN values.")
        # 检查Fai_2是否有NaN值
        if torch.any(torch.isnan(Fai_2)):
            print("Warning: Fai_2 contains NaN values.")
        # 检查Fai_3是否有NaN值
        if torch.any(torch.isnan(Fai_3)):
            print("Warning: Fai_3 contains NaN values.")

        # 计算omiga和sigma
        Alpha = torch.div(torch.mul(Bright_Temp - torch.mul(alpha_K, L_lambda), L_lambda), Bright_Temp)

        Beta = torch.div(torch.mul(Beta_K, L_lambda.pow(2)), Bright_Temp.pow(2))

        # 检查Alpha是否有NaN值
        if torch.any(torch.isnan(Alpha)):
            print("Warning: Alpha contains NaN values.")

        # 检查Beta是否有NaN值
        if torch.any(torch.isnan(Beta)):
            print("Warning: Beta contains NaN values.")

        Omiga = torch.div(1, Beta)  # 防止除零
        Sigma = -torch.div(Alpha, Beta)  # 防止除零

        # 检查Omiga是否有NaN值
        if torch.any(torch.isnan(Omiga)):
            print("Warning: Omiga contains NaN values.")

        # 检查Sigma是否有NaN值
        if torch.any(torch.isnan(Sigma)):
            print("Warning: Sigma contains NaN values.")

        # 代入辐射传输方程，计算出地物真实地表辐射亮度
        Ts = torch.mul(Omiga, torch.div(torch.mul(Fai_1, L_lambda) + Fai_2, surface_emiss) + Fai_3) + Sigma

        # 检查Ts中是否有NaN值
        if torch.any(torch.isnan(Ts)):
            print("Warning: Ts contains NaN values.")

        attention = self.conv(Ts)

        return attention