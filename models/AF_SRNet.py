import torch
import torch.nn as nn
import torch.nn.functional as F

class TemporalModule(nn.Module):
    def __init__(self, feature_size):
        super(TemporalModule, self).__init__()
        self.W_TE = nn.Linear(feature_size, feature_size)
        self.W_T_prev = nn.Linear(feature_size, feature_size)
        # Temporal attention mechanism can be more complex based on actual implementation
        self.temporal_attention = nn.Sequential(
            nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1),
            nn.Tanh()
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, TE, T_prev, TR_pre):
        AT = self.sigmoid(self.W_TE(TE) + self.W_T_prev(T_prev))
        T = torch.tanh(self.W_TE(TE) + self.W_T_prev(T_prev))
        ATT_T = self.temporal_attention(TR_pre)  # Assuming TR_pre is an attention-weighted state
        T_output = (T + ATT_T) * AT
        return T_output


class SpatialModule(nn.Module):
    def __init__(self, feature_size):
        super(SpatialModule, self).__init__()
        self.W_SE = nn.Linear(feature_size, feature_size)
        self.W_S_prev = nn.Linear(feature_size, feature_size)
        # Spatial attention mechanism
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1),
            nn.Tanh()
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, SE, S_prev, SR_pre):
        AS = self.sigmoid(self.W_SE(SE) + self.W_S_prev(S_prev))
        S = torch.tanh(self.W_SE(SE) + self.W_S_prev(S_prev))
        ATT_S = self.spatial_attention(SR_pre)  # Assuming SR_pre is an attention-weighted state
        S_output = (S + ATT_S) * AS
        return S_output


class SpatiotemporalResidualUnit(nn.Module):
    def __init__(self, temporal_feature_size, spatial_feature_size):
        super(SpatiotemporalResidualUnit, self).__init__()
        self.temporal_module = TemporalModule(temporal_feature_size)
        self.spatial_module = SpatialModule(spatial_feature_size)
        self.W_ST_SE = nn.Linear(spatial_feature_size, spatial_feature_size)
        self.W_T = nn.Linear(temporal_feature_size, temporal_feature_size)
        self.W_S = nn.Linear(spatial_feature_size, spatial_feature_size)
        self.conv_1x1 = nn.Conv2d(temporal_feature_size + spatial_feature_size, temporal_feature_size + spatial_feature_size, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, TE, SE, STE, T_prev, S_prev):
        T = self.temporal_module(TE, T_prev)
        S = self.spatial_module(SE, S_prev)

        AST = self.sigmoid(self.W_ST_SE(STE) + self.W_T(T) + self.W_S(S))
        ST = self.conv_1x1(torch.cat([TE, SE], dim=1))
        STR = AST * torch.tanh(self.conv_1x1(torch.cat([T, S], dim=1)))
        H = ST + STR
        return H


# class Encoder(nn.Module):
#     def __init__(self, feature_size):
#         super(Encoder, self).__init__()
#         self.sru1 = SpatiotemporalResidualUnit(feature_size, feature_size)
#         self.sru2 = SpatiotemporalResidualUnit(feature_size, feature_size)
#         self.sru3 = SpatiotemporalResidualUnit(feature_size, feature_size)
#         self.sru4 = SpatiotemporalResidualUnit(feature_size, feature_size)

#     def forward(self, H0):
#         H1 = self.sru1(H0, H0, H0, H0, H0)
#         H2 = self.sru2(H1, H1, H1, H1, H1)
#         H3 = self.sru3(H2, H2, H2, H2, H2)
#         H4 = self.sru4(H3, H3, H3, H3, H3)

#         return H4

class Encoder(nn.Module):
    def __init__(self, feature_size, num_units=4):
        super(Encoder, self).__init__()
        self.num_units = num_units
        self.sru_units = nn.ModuleList([
            SpatiotemporalResidualUnit(feature_size, feature_size)
            for _ in range(num_units)
        ])

    def forward(self, H0):
        H = H0
        for unit in self.sru_units:
            H = unit(H)
        return H


class AttentionFusionBlock(nn.Module):
    def __init__(self, channel_size):
        super(AttentionFusionBlock, self).__init__()
        # Global Average Pooling layer
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Point-wise convolutional layers to transform the local features
        self.pointwise_conv_local_1 = nn.Conv2d(channel_size, channel_size, kernel_size=1)
        self.pointwise_conv_local_2 = nn.Conv2d(channel_size, channel_size, kernel_size=1)

        # Point-wise convolutional layers to transform the global features
        self.pointwise_conv_global_1 = nn.Conv2d(channel_size, channel_size, kernel_size=1)
        self.pointwise_conv_global_2 = nn.Conv2d(channel_size, channel_size, kernel_size=1)
        
        # ReLU activation function
        self.relu = nn.ReLU(inplace=True)
        
        # Sigmoid activation function for generating attention weights
        self.sigmoid = nn.Sigmoid()

    def forward(self, H_radar, H_precip):

        H = H_radar + H_precip

        # Perform global average pooling on (H_radar \oplus H_precip)
        H_pooling = self.global_avg_pool(H)

        # Perform point-wise convolution
        H_local = self.pointwise_conv_local_1(H)
        H_global = self.pointwise_conv_global_1(H_pooling)

        # Apply sigmoid function
        H_local = self.relu(H_local)
        H_global = self.relu(H_global)

        # Perform point-wise convolution
        H_local = self.pointwise_conv_local_2(H_local)
        H_global = self.pointwise_conv_global_2(H_global)
        
        # Apply sigmoid function to get attention weights
        attention_weights = self.sigmoid(H_local + H_global)
        
        # Apply attention weights to radar features and inverse attention to precip features
        H_fusion = attention_weights * H_radar + (1 - attention_weights) * H_precip
        
        return H_fusion


class Decoder(nn.Module):
    def __init__(self, feature_size, num_units=4):
        super(Decoder, self).__init__()
        # We're using the same SpatiotemporalResidualUnit as in the Encoder
        self.num_units = num_units
        self.sru_units = nn.ModuleList([
            SpatiotemporalResidualUnit(feature_size, feature_size)
            for _ in range(num_units)
        ])
        # Additionally, we might need some upsampling layers if we want to increase
        # the spatial dimensions of our feature maps
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, H_fusion):
        H = H_fusion
        for unit in self.sru_units:
            # Apply each SRU unit
            H = unit(H)
            # Upsample the output feature map to increase its spatial dimensions
            H = self.upsample(H)
        return H


# 定义AF-SRNet模型
class AFSRNet(nn.Module):
    def __init__(self, feature_size):
        super(AFSRNet, self).__init__()
        self.encoder_radar = Encoder(feature_size)
        self.encoder_precip = Encoder(feature_size)
        self.attention_fusion = AttentionFusionBlock(feature_size)
        self.decoder = Decoder(feature_size)

    def forward(self, radar_data, precip_data):
        radar_feature = self.encoder_radar(radar_data)
        precip_feature = self.encoder_precip(precip_data)
        fused_feature = self.attention_fusion(radar_feature, precip_feature)
        output = self.decoder(fused_feature)
        return output

# # 实例化模型
# af_srnet_model = AFSRNet()

# # 假设输入数据
# radar_data_sample = torch.randn(1, 3, 64, 64)  # 示例雷达数据
# precip_data_sample = torch.randn(1, 3, 64, 64)  # 示例降水数据

# # 前向传播
# output = af_srnet_model(radar_data_sample, precip_data_sample)
