
from alphagen_qlib.stock_data import StockData
def get_data_my(instru,start,end,raw=False,qlib_path = '',freq = 'day'):
    import qlib
    from qlib.data import D
    qlib.init(provider_uri=qlib_path, region='cn')
    def get_instruments(name,start,end):
        instru =  D.instruments(name)
        return D.list_instruments(
            instruments=instru, 
            start_time=start, 
            end_time=end, 
            as_list=True,
            freq=freq,
            
        )
    instru = get_instruments(instru,start,end)
    return StockData(instru,start,end,raw = raw,qlib_path = qlib_path,freq = freq)