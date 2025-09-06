from numba import cuda, float32, int16, int32, float64
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_normal_float32
import math
# 计算标准结构的PV的cuda函数
@cuda.jit
def classical_cuda_kernal(
    rng_states,
    nsim_per_thread, 
    s,
    k,
    ki,
    floor,
    vol,
    r,
    tday,
    ki_flag_existed,

    final_tday,
    final_rebate_dis,
    final_dis_factor,

    cpn_idx_init,
    cpn_amt_dis,
    cpn_barrier,
    cpn_tday,

    call_idx_init,
    call_amt_dis,
    call_barrier,
    call_tday, 

    forward_div,

    margin,
    dis_factor,
    enhance_k,          # 增强执行价
    enhance_ratio,
    option_end_cost,    # 年化中收 * 期初价
    ki_expire,          # 是否为到期观察敲入 
    callput,               # 是否看涨
    out
):
    
    thread_id = cuda.grid(1)
    i = int16(0)
    t = int16(0)
    cpn_idx = int16(0)
    call_idx = int16(0)
    st = float32(0)
    pv_sum = float32(0)
    dt = float32(1/244)
    rand_arg1 = float32(math.exp(-0.5* vol**2 *dt))
    rand_arg2 = float32(vol*math.sqrt(dt))
    
    if callput:  # 看涨
        for i in range(nsim_per_thread):
            # 初始化参数
            st = s
            cpn_idx = cpn_idx_init
            call_idx = call_idx_init
            ki_flag = ki_flag
            pv = 0
            
            for t in range(tday, final_tday+1):
                # 判断是否敲入
                ki_flag = ki_flag_existed or (st<=ki)
                
                # 计息日
                if t == cpn_tday[cpn_idx]:
                    if st>= cpn_barrier[cpn_idx]:
                        pv += cpn_amt_dis[cpn_idx]
                    cpn_idx += 1
                
                # 提前赎回日
                if t == call_tday[call_idx]:
                    if st >= call_barrier[call_idx]:
                        pv += call_amt_dis[call_idx] + dis_factor[call_idx] * (max(st-enhance_k, 0) * enhance_ratio + margin)
                        break
                    call_idx += 1
                    
                # 到期日
                if t == final_tday:
                    if ki_expire:   # 到期观察敲入，欧式雪球
                        if st <= ki:
                            pv += final_dis_factor * max(min(st-k, 0), min(floor-k,0))
                        else:
                            pv += final_rebate_dis  
                    else:     
                        if ki_flag:
                            pv += final_dis_factor * max(min(st-k, 0), min(floor-k,0))
                        else:
                            pv += final_rebate_dis  
                            
                    pv += final_dis_factor * margin
                    break
            
                # 生成次日价格
                rand = xoroshiro128p_normal_float32(rng_states, thread_id)  # 生成随机数
                st *= forward_div[t-tday] * rand_arg1 * math.exp(rand_arg2 * rand)
            pv_sum += pv
            pv_sum += option_end_cost*(t-tday)/244 * math.exp(-r*(t-tday)/244)  # 计算后端返息
            
    else:
        # 看跌
        
        for i in range(nsim_per_thread):
        # 初始化参数
            st = s
            cpn_idx = cpn_idx_init
            call_idx = call_idx_init
            ki_flag = ki_flag
            pv = 0
            
            for t in range(tday, final_tday+1):
                # 判断是否敲入
                ki_flag = ki_flag_existed or (st>=ki)
                
                # 计息日
                if t == cpn_tday[cpn_idx]:
                    if st<= cpn_barrier[cpn_idx]:
                        pv += cpn_amt_dis[cpn_idx]
                    cpn_idx += 1
                
                # 提前赎回日
                if t == call_tday[call_idx]:
                    if st <= call_barrier[call_idx]:
                        pv += call_amt_dis[call_idx] + dis_factor[call_idx] * (max(enhance_k - st, 0) * enhance_ratio + margin)
                        break
                    call_idx += 1
                    
                # 到期日
                if t == final_tday:
                    
                    if ki_expire:   # 到期观察敲入，欧式雪球
                        if st >= ki:
                            pv += final_dis_factor * max(min(k-st, 0), min(k-floor,0))
                        else:
                            pv += final_rebate_dis  
                    else:     
                        if ki_flag:
                            pv += final_dis_factor * max(min(k-st, 0), min(k-floor,0))
                        else:
                            pv += final_rebate_dis  
                            
                    pv += final_dis_factor * margin
                    break
            
                # 生成次日价格
                rand = xoroshiro128p_normal_float32(rng_states, thread_id)
                st *= forward_div[t-tday] * rand_arg1 * math.exp(rand_arg2 * rand)
            
            pv_sum += pv
            pv_sum += option_end_cost*(t-tday)/244 * math.exp(-r*(t-tday)/244)  # 考虑未支付中收
            
    out[thread_id] = pv_sum / nsim_per_thread - margin

# 计算折价建仓的PV的cuda函数
@cuda.jit
def discounted_cuda_kernal(
        rng_states,
        nsim_per_thread,
        s0,
        s,
        k,
        ki,
        floor,
        vol,
        r,
        tday,
        ki_flag_existed,
        final_tday,
        final_rebate_dis,
        final_dis_factor,
        cpn_idx_init,
        cpn_amt_dis,
        cpn_barrier,
        cpn_tday,
        call_idx_init,
        call_amt_dis,
        call_barrier,
        call_tday,
        forward_div,
        dis_factor,
        margin,
        option_end_cost,
        out
    ):

    thread_id = cuda.grid(1)

    i = int16(0)
    t = int16(0)
    cpn_idx = int16(0)
    call_idx = int16(0)
    st = float32(0)
    pv_sum = float32(0)
    dt = float32(1/244)
    rand_arg1 = float32(math.exp(-0.5 * vol**2 * dt))
    rand_arg2 = float32(vol*math.sqrt(dt))

    for i in range(nsim_per_thread):

        # 初始化参数
        st = s
        cpn_idx = cpn_idx_init
        call_idx = call_idx_init
        ki_flag = ki_flag_existed
        pv = 0

        for t in range(tday, final_tday+1):

            # 判断是否敲入
            ki_flag = ki_flag or (st <= ki)

            # 计息日
            if t == cpn_tday[cpn_idx]:
                if st >= cpn_barrier[cpn_idx]:
                    pv += cpn_amt_dis[cpn_idx]
                cpn_idx += 1

            # 提前赎回日
            if t == call_tday[call_idx]:
                if st >= call_barrier[call_idx] and not ki_flag:
                    pv += call_amt_dis[call_idx] + dis_factor[call_idx] * margin
                    break
                call_idx += 1

            # 到期日
            if t == final_tday:
                if ki_flag:
                    pv += final_dis_factor * (st-k) * s0/k   # 敲入后折价建仓收益
                else:
                    pv += final_rebate_dis  # 未敲入未敲出收益
                
                pv += final_dis_factor * margin
                break

            # 生成次日价格
            rand = xoroshiro128p_normal_float32(rng_states, thread_id)
            st *= forward_div[t-tday] * rand_arg1 * math.exp(rand_arg2 * rand)

        pv_sum += pv
        pv_sum += option_end_cost*(t-tday)/244 * math.exp(-r*(t-tday)/244)  # 考虑未支付中收
    out[thread_id] = pv_sum / nsim_per_thread - margin

# 计算FCN结构PV的cuda函数   
@cuda.jit
def fcn_cuda_kernal(
        rng_states, 
        nsim_per_thread,
        s0,
        s,
        k,
        ki,
        vol,
        r,
        tday,
        final_tday,
        final_rebate_dis,
        final_dis_factor,
        call_idx_init,
        call_amt_dis,
        call_barrier,
        call_tday,
        forward_div,
        dis_factor,
        margin,
        option_end_cost,
        out
    ):

    thread_id = cuda.grid(1)

    i = int16(0)
    t = int16(0)
    call_idx = int16(0)
    st = float32(0)
    pv_sum = float32(0)
    dt = float32(1/244)
    rand_arg1 = float32(math.exp(-0.5 * vol**2 * dt))
    rand_arg2 = float32(vol*math.sqrt(dt))

    for i in range(nsim_per_thread):

        # 初始化参数
        st = s
        call_idx = call_idx_init
        pv = 0

        for t in range(tday, final_tday+1):

            # 提前赎回日
            if t == call_tday[call_idx]:
                if st >= call_barrier[call_idx]:
                    pv += call_amt_dis[call_idx] + dis_factor[call_idx] * margin
                    break
                call_idx += 1

            # 到期日
            if t == final_tday:
                if st < ki:
                    pv += final_dis_factor * (st-k) * s0/k
                pv += final_rebate_dis
                pv += final_dis_factor * margin
                break

            # 生成次日价格
            rand = xoroshiro128p_normal_float32(rng_states, thread_id)
            st *= forward_div[t-tday] * rand_arg1 * math.exp(rand_arg2 * rand)

        pv_sum += pv
        pv_sum += option_end_cost*(t-tday)/244 * math.exp(-r*(t-tday)/244)  # 考虑未支付中收
        
    out[thread_id] = pv_sum / nsim_per_thread - margin

