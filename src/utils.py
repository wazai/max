"""
Utility functions

@author: jingweiwu
"""

def date_10to8(date):
    return date[:4] + date[5:7] + date[8:10]

def date_8to10(date):
    return date[:4] + '-' + date[4:6] + '-' + date[6:8]