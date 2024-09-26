import sys, os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import numpy as np
from common.util import im2col



class Convolution :
    def __init__(self, W, b, stride=1, pad=0):
        self.W = W # 가중치 (필터), 형상은 (FN, C, FH, FW)
        self.b = b # 편향
        self.stride = stride # 스트라이드
        self.pad = pad   # 패딩
    
    def forward(self, x):
        FN, C, FH, FW = self.W.shape
        N, C, H, W = x.shape # 입력 데이터의 형상
        
        # 출력 데이터의 형상 계산
        out_h = int(1 + (H + 2*self.pad - FH) / self.stride) 
        out_w = int(1 + (W + 2*self.pad - FW) / self.stride) 
        
        # 입력 데이터를 im2col로 전개
        col = im2col(x, FH, FW, self.stride, self.pad)
        
        # 필터를 reshape
        col_W = self.W.reshape(FN, -1).T 
        # -1 다차원 배열의 원소 수가 변환 후에도 똑같이 유지되도록 적절히 묶어줌
        # 예를 들어 (10, 2, 3, 2) -> (10, 12)로 변환
        
        # 행렬의 내적을 계산하고 편향을 더함
        out = np.dot(col, col_W) + self.b
        
        # 출력 데이터를 reshape
        out = out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2)
        # (N, H, W, C) -> (N, C, H, W) 축 순서 변경
        return out