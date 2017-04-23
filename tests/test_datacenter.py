import src.datacenter as dc

d = dc.DataCenter()
df = d.load_codes_return(['000001', '000002'], '2010-01-01', '2010-02-01')
print(df.head())


