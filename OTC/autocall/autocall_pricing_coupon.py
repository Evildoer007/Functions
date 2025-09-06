import autocall_pricing_code



def batch_price(
        target='coupon',
        args_list=[], 
        mold='snowball', 
        delta=False, 
        gamma=False,
        vega=False, 
        theta=False,
        k=100, 
        outfile=None, 
        call_freq=None, 
        forward=None, 
        r=0.03, 
        ki_expire=False,
        discount=False, 
        fcn=False, 
        parachute=False,
    ):

    count = 0
    rslt = []
    for args in args_list:
        ki = args[0]
        ko = args[1]
        floor = args[2]
        ko_step_down = args[3]
        tenor_month = args[4]
        call_begin_month = args[5]
        final_rebate_ann = args[6]
        margin_ratio = args[7]
        vol = args[8]/100
        b = args[9]
        pv_coupon = args[10]   #计算票息时输入pv，计算pv时输入票息
        if len(args) >= 12:
            last_month=args[11]
            last_coupon=args[12]
        else:
            last_month=None
            last_coupon=None
        
        q = r-b/100
        call_tday=None
        margin=100*margin_ratio

        trade = autocall_pricing_code.autocall(
            mold=mold,
            s=100,
            ki=ki,
            floor=floor,
            ko=ko,
            ko_step_down=ko_step_down,
            vol=vol,
            r=r,
            q=q,
            tenor_month=tenor_month,
            call_begin_month=call_begin_month,
            call_tday=call_tday,
            coupon=pv_coupon,
            tday=1,
            ki_flag=False,
            nsim=autocall_pricing_code.NSIM,
            s0=100,
            k=k,
            forward=forward,
            margin = margin,
            # parachute=True,    # 降落伞
            # call_barrier = [100]*5+[70]*1
        )            
        if final_rebate_ann is not None:
            trade.final_rebate=final_rebate_ann  * trade.final_nday / 365
            
            
        if obs_freq == 'monthly':
            if mold =='trigger':
                call_tday=[round((x+1)/12*244) + round((call_begin_month-1)/12*244)
                              for x in range(tenor_month-call_begin_month+1)]
                trade = autocall(
                    mold=mold,
                    s=100,  # 现价
                    ki=ki,  # 敲入价
                    floor=floor,   # 保底价#
                    ko=ko,  # 敲出价
                    ko_step_down=ko_step_down,   # 敲出递减步长
                    vol=vol,  # 波动率
                    r=r,  # 贴现率
                    q=q,
                    tenor_month=tenor_month,
                    call_begin_month=call_begin_month,
                    call_tday=call_tday,
                    call_nday= [round(x/244*365) for x in call_tday],
                    coupon=pv_coupon,  # +1.7,
                    tday=1,   # 第n个交易日，起始为1
                    ki_flag=False,    # 敲入标记
                    nsim=NSIM,
                    s0=100,
                    k=k,  # 执行价
                    forward=forward,
                    margin = margin,
                    # parachute=True,    # 降落伞
                    # call_barrier = [100]*9+[70]*1,#[103,103,103,103,103,103,103,103,103,103,103,90], 
                )                
                
                if final_rebate_ann is not None:
                    trade.final_rebate=final_rebate_ann  * trade.final_nday / 365


        print(trade.mold + " %d/%d %dm/%dm" % (trade.ki, trade.ko,
              trade.tenor_month, trade.call_begin_month) + ' floor%d' % trade.floor)
        print("vol: %.2f%%, b: %.2f%%" %
              (trade.vol*100, (trade.r-trade.q)*100))
        print("call_barrier: ", [(i+trade.call_begin_month, trade.call_barrier[i])
              for i in range(len(trade.call_barrier))])
        
        count += 1
        print('finished %d%%, time elapsed %.2f s' % (count/len(args_list)*100, time.time()-t0))
        
        if target=='coupon':
            trade.coupon = trade.faircoupon(pv_target=pv_coupon, final_rebate=final_rebate_ann, last_month=last_month, last_coupon=last_coupon)
            print("pv: %.2f" % pv_coupon)
            if last_month is None:
                print("红利票息: %.2f" % (trade.final_rebate / trade.final_nday * 365))
                print("敲出票息: %.2f" % trade.coupon)            
            else:
                print("早利雪球")
                print("票息后%d期: %.2f" % (len(trade.call_amt)- last_month, last_coupon))
                print("票息前%d期: %.2f" % (last_month, trade.coupon))

        else:
            pv = trade.price()
            print("红利票息: %.2f" % (trade.final_rebate / trade.final_nday * 365))
            print("敲出票息: %.2f" % trade.coupon)            
            print("pv: %.2f" % pv)
        

        vega0 = 0
        if vega:
            vega0 = trade.vega()
            print("vega %.4f, vega(100w)%.4f" % (vega0, vega0*10000))
        delta0 = 0
        if delta:
            delta0 = trade.delta()
            print("delta %.4f" % delta0)
        print("------------")
        
        rslt0 ={'ki': trade.ki,
                'ko': trade.ko,
                'ko_step': ko_step_down,
                'tenor': trade.tenor_month,
                'vol': trade.vol,
                'b': trade.r - trade.q,
                'fwd': forward,
                'vega': vega0 * 10000,
                'delta': delta0,
                }
        if target=='coupon':
            rslt0['coupon'] = trade.coupon/100
        else:
            rslt0['pv'] = pv
        rslt.append(rslt0)
    rslt = pd.DataFrame(rslt)
    if outfile is None:
        rslt.to_excel('snowball_coupon_rslt.xlsx', index=False)
    else:
        rslt.to_excel(outfile, index=False)
        
    print('============')
    if target=='coupon':
        print(rslt['coupon']*100)
        print(rslt['vega'])
    elif target == 'price':
        print(rslt['pv'])

    return rslt








price_result = batch_price(
    args_list,
    forward = fwd,
    r = 0.03,
    mold = 'snowball',
    delta = True,
    gamma = True,
    vega = True,
    theta = true,
    k = 100,
    call_freq = 'monthly',
    target = 'coupon',
)