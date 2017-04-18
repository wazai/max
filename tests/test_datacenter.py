import src.datacenter as dc

d = dc.DataCenter()
df = d.load_codes_return(['000001','000002'],'20100101','20100201')
print(df.head())


