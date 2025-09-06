from __future__ import print_function, absolute_import
import numpy as np
import pandas as pd
import time
import sys
import copy
import scipy.optimize as opt
import math
import cuda_kernel_package
from numba import cuda, float32, int16, int32, float64
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_normal_float32

NSIM = int(1e7)
Annual_trade_days = 244
class autocall:
    def __init__(
            self,

            target='coupon',        # 测算目标：coupon票息pv估值
            # 基本结构要素
            # ↓↓↓
            s0=None,                # 期初价格
            s=None,                 # 标的现价
            k=None,                 # 执行价
            ki=None,                # 敲入价
            floor=0,                # 保底价
            margin=0,               # 保证金，全保为s0
            ko=None,                # 敲出价
            ko_step_down=None,      # 敲出价递减步长
            vol=0,                  # 波动率
            tenor_month=12,         # 期限/月
            call_freq="monthly",    # 观察频率monthly月观察biweekly双周观察weekly周观察
            call_begin_month=1,     # 敲出观察起始周期/锁定期
            mold='snowball',        # 类型snowball雪球, phoenix凤凰, trigger触发器
            option_end_cost=0,      # 后端年化返息
            option_must_cost=0,     # 后端绝对返息
            startdate=None,         # 起始日
            valuedate=None,         # 估值日

            # 估值参数要素
            # ↓↓↓
            tday=1,                 # 第n个交易日，初始1
            nday=None,              # 第n个自然日，初始1
            ki_flag=False,          # 已敲入True, 未敲入False
            r=0.04,                 # 贴现率
            q=0,                    # 股息率
            coupon=None,            # 票息
            coupon_list=None,       # 票息列表
            call_num=None,          # 敲出次数
            call_tday=None,         # 敲出观察日，交易日列表
            call_nday=None,         # 敲出观察日，自然日列表，用于计算贴现
            call_amt=None,          # 敲出金额
            call_barrier=None,      # 敲出障碍
            cpn_num=None,           # 计息次数
            cpn_amt=None,           # 计息金额
            cpn_barrier=None,       # 计息障碍
            cpn_tday=None,          # 计息观察日，交易日列表
            cpn_nday=None,          # 计息观察日，自然日列表
            final_tday=None,        # 到期日，交易日
            final_nday=None,        # 到期日，自然日
            final_rebate=None,      # 到期收益
            forward=None,           # 远期价格，格式[[fwd1, tday1],..., [fwdn, tdayn]]
            forward_ratio=1,        # 远期曲线比例，例如比例0.3，年化贴水10%将被调整为3%
            nsim=10000000,          # 蒙卡次数
            dt=1/Annual_trade_days, # 交易日差分

            # 期权类型要素
            # ↓↓↓
            callput='call',         # 看涨call/看跌put
            enhance_k=None,         # 增强雪球的增强执行价，默认等于敲出价
            enhance_ratio=0,        # 增强雪球的参与率
            parachute=False,        # 降落伞结构True，最后一期敲出价=敲入价
            discount=False,         # 折价建仓结构为True
            fcn=False,              # fcn结构为True
            ki_expire=False,        # 欧式结构为True(到期观察敲入)
        ):
        
        # 赋值要素
        self.s0 = s0
        self.s = s
        self.k = k
        self.ki = ki
        self.floor = floor
        self.margin = margin
        self.ko = ko
        self.vol = vol
        self.tenor_month = tenor_month
        self.call_freq = call_freq
        self.call_begin_month = call_begin_month
        self.mold = mold
        self.option_end_cost = option_end_cost
        self.option_must_cost = option_must_cost
        self.startdate = startdate
        self.valuedate = valuedate

        self.tday = tday
        self.nday = round(tday/244*365) if nday is None else nday
        self.ki_flag = 1 if ki_flag else 0
        self.r = r
        self.q = q
        self.coupon = coupon
        self.coupon_list=coupon_list
        self.nsim = nsim
        self.dt = dt

        self.callput = callput
        self.enhance_k = 999999 if enhance_k is None else enhance_k
        self.enhance_ratio = enhance_ratio

        # 处理敲出递减序列
        if ko_step_down is not None and call_barrier is None:
            if ko is None:
                print('Attention!!! No Knock-Out price!!!')
            call_barrier = [ ko + x * ko_step_down for x in range(tenor_month-call_begin_month+1) ]


        if target == 'coupon':
            # 处理日期数据
            if call_tday is None:
                if mold == 'trigger': # 触发器绝对票息，周观察
                    self.call_tday = [ (x+1) * 5 for x in range(4*tenor_month) ]
                elif call_freq == 'weekly':
                    call_tday = [ (x+1) * 5 for x in range((tenor_month-call_begin_month) * 4) ] if call_begin_month == 0 else [ 
                        round(call_begin_month/12 * 244) + 5 * x for x in range((tenor_month-call_begin_month)*4+1) ]
                    tenor_month = tenor_month * 4
                    call_begin_month = call_begin_month * 4
                elif call_freq == 'biweekly':
                    call_tday = [ (x+1) * 10 for x in range((tenor_month-call_begin_month) * 2) ] if call_begin_month == 0 else [
                        round(call_begin_month/12 * 244) + 10 * x for x in range((tenor_month-call_begin_month)*2+1) ]
                    tenor_month = tenor_month * 2
                    call_begin_month = call_begin_month * 2
                elif call_freq == 'monthly':
                    self.call_tday = [ round((x+1)/12 * 244) + round((call_begin_month-1)/12 * 244)
                                    for x in range(tenor_month-call_begin_month+1) ]
            else:
                self.call_tday = call_tday
                    
            self.call_num = len(self.call_tday) if call_num is None else call_num
            self.call_barrier = [ko] * self.call_num if call_barrier is None else call_barrier

            if call_nday is None:
                if mold == 'trigger':
                    self.call_nday = [ round(x/5*7) for x in self.call_tday ]
                else:
                    self.call_nday = [ round(x/244*365) for x in self.call_tday ]  
            else:
                self.call_nday = call_nday

            if cpn_tday is None:
                if mold == 'phoenix':
                    self.cpn_tday = [(round((x+1)/12*244)) for x in range(tenor_month)]      # 凤凰每月计息
                else:
                    self.cpn_tday = self.call_tday
            else:
                self.cpn_tday = cpn_tday
            self.cpn_nday = [round(x/244*365) for x in self.cpn_tday] if cpn_nday is None else cpn_nday
            self.cpn_num = len(self.cpn_tday) if cpn_num is None else cpn_num
            self.cpn_barrier = [ki] * self.cpn_num if cpn_barrier is None else cpn_barrier    # 计息障碍默认为敲入线

            self.final_tday = self.call_tday[-1] if final_tday is None else final_tday
            self.final_nday = self.call_nday[-1] if final_nday is None else final_nday
        else:
            pass
            # ========未完成========
            #self.call_tday = self.calc_tday()
            #self.call_nday = self.calc_nday()
            #self.call_num = len(self.call_tday)

        # 处理票息金额
        self.call_amt = call_amt
        self.cpn_amt = cpn_amt
        self.final_rebate = final_rebate if final_rebate is None else final_rebate * self.call_nday[-1]/365 # 年化
        if self.mold == 'snowball':
            if self.call_amt is None:
                if self.coupon_list is None:
                    self.call_amt = [self.coupon * x / 365 for x in self.call_nday]      # 自然日/365计息
                else:
                    self.call_amt = [x * y / 365 for x, y in zip(self.coupon_list, self.call_nday)]   
            if self.cpn_amt is None:
                self.cpn_amt = [0] * self.cpn_num
            if self.final_rebate is None:
                if self.coupon_list is None:
                    self.final_rebate = self.coupon * self.final_nday / 365
                else:
                    self.final_rebate = self.call_amt[-1]      
        elif self.mold == 'phoenix':
            if self.call_amt is None:
                self.call_amt = [0] * self.call_num
            if self.cpn_amt is None:
                self.cpn_amt = [self.coupon/12] * self.cpn_num     # 月/12计息
            if self.final_rebate is None:
                self.final_rebate = 0
        elif self.mold == 'trigger':
            if self.call_amt is None:
                self.call_amt = [self.coupon] * self.call_num     # 绝对票息
            if self.cpn_amt is None:
                self.cpn_amt = [0] * self.cpn_num
            if self.final_rebate is None:
                self.final_rebate = self.coupon
        # 降落伞结构最后一期敲出等于敲入
        if parachute: self.call_barrier[-1] = self.ki

        self.forward_base = forward
        self.forward = None if forward is None else [[s+(x[0]-s)*forward_ratio, x[1]] for x in forward]
        self.forward_ratio = forward_ratio

        self.discount = 1 if discount else 0
        self.fcn = 1 if fcn else 0
        self.ki_expire = 1 if ki_expire else 0

    def price(self, k = None, vol = None, s = None, forward_curve = None):

        s = self.s if s is None else s
        k = self.k if k is None else k
        vol = self.vol if vol is None else vol
        forward_curve = self.forward_curve if forward_curve is None else forward_curve

        threads_per_block = 64
        blocks = 512
        threads = threads_per_block * blocks
        nsim_per_thread = int32(self.nsim // threads)
        rng_states = create_xoroshiro128p_states(threads, seed=1) # 随机种子
        out = np.zeros(threads, dtype=np.float32)

        call_tday = np.array(self.call_tday)
        cpn_tday = np.array(self.cpn_tday)

        if self.tday >= call_tday[-1]:      # 是否到期日
            call_idx_init = len(call_tday) - 1
            cpn_idx_init = len(call_tday) - 1
        else:
            # 下个call_tday
            call_idx_init = np.where(self.tday <= call_tday)[0][0].astype(np.int16)   
            # 下个cpn_tday
            cpn_idx_init = np.where(self.tday <= cpn_tday)[0][0].astype(np.int16)

        # 贴现cpn_amt
        cpn_amt_dis = [self.cpn_amt[i] * np.exp(-self.r * (
            self.cpn_nday[i] - self.nday)/365) for i in range(self.cpn_num)]
        # 贴现call_amt(考虑保证金) 
        call_amt_dis = [(self.call_amt[i]+self.margin) * np.exp(-self.r * (
            self.call_nday[i] - self.nday)/365) - self.margin for i in range(self.call_num)]  
        
        # 敲出贴现因子
        dis_factor = [np.exp(-self.r * (self.call_nday[i] - self.nday)/365) for i in range(self.call_num)]

        # 期末贴现因子
        final_dis_factor = np.exp(-self.r * (self.final_nday - self.nday)/365)
        # 期末贴现赔付(考虑保证金)
        final_rebate_dis = final_dis_factor * (self.final_rebate + self.margin) - self.margin
 
        forward_curve_array = np.array(self.forward_curve if forward_curve is None else forward_curve).astype(np.float32)
        forward_curve_div = forward_curve_array[1:] / forward_curve_array[:-1]      # forward明日/当日

        if self.discount:
            pass
        elif self.fcn:
            pass
        elif self.enhance:
            pass
        else:
            cuda_kernal[blocks, threads_per_block](
                rng_states,
                nsim_per_thread,
                float32(self.s if s is None else s),
                float32(self.k if k is None else k),
                float32(self.ki),
                float32(self.floor),
                float32(self.vol if vol is None else vol),
                float32(self.r),
                int16(self.tday),
                self.ki_flag,

                int16(self.final_tday),
                float32(final_rebate_dis),
                float32(final_dis_factor),

                int16(cpn_idx_init),
                np.array(cpn_amt_dis).astype(np.float32),
                np.array(self.cpn_barrier).astype(np.float32),
                np.array(self.cpn_tday).astype(np.int16),

                int16(call_idx_init),
                np.array(call_amt_dis).astype(np.float32),
                np.array(self.call_barrier).astype(np.float32),
                np.array(self.call_tday).astype(np.int16),

                forward_curve_div,

                float32(self.margin),
                np.array(dis_factor).astype(np.float32),
                float32(self.enhance_k),
                float32(self.enhance_ratio),
                float32(self.option_end_cost)
                self.ki_expire,
                self.callput,
                out,
            )

        self.pv = np.mean(out)
        return self.pv
        
