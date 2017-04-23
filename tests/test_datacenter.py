import src.datacenter as dc
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s  %(name)s  %(levelname)s  %(message)s')

d = dc.DataCenter()
df = d.load_codes_return(['000001', '000002'], '2010-01-01', '2010-02-01')
print(df.head())


