# -*- coding: utf8 -*-
import pandas
import names


data = pandas.DataFrame([
    ['2006-09-01',	4.2260,	4.9000,	1.0490,	14.0000,],
    ['2006-10-01',	4.0120,	5.0400,	1.0470,	13.9000,],
    ['2006-11-01',	4.4270,	5.9000,	1.0440,	13.8000,],
    ['2006-12-01',	4.7670,	5.9200,	1.0380,	13.8000,],
    ['2007-01-01',	4.9150,	5.7500,	1.0290,	13.7000,],
    ['2007-02-01',	4.6650,	5.6100,	1.0230,	13.5500,],
    ['2007-03-01',	4.9200,	5.7800,	1.0170,	13.5500,],
    ['2007-04-01',	4.7750,	5.7500,	1.0130,	13.4000,],
    ['2007-05-01',	4.6930,	5.4500,	1.0110,	13.2000,],
    ['2007-06-01',	4.5800,	5.1700,	1.0100,	13.2000,],
    ['2007-07-01',	4.2730,	5.1600,	1.0120,	13.0000,],
    ['2007-08-01',	4.3720,	5.6500,	1.0130,	12.8500,],
    ['2007-09-01',	5.2270,	6.7700,	1.0140,	12.8500,],
    ['2007-10-01',	4.7340,	7.1600,	1.0140,	12.7000,],
    ['2007-11-01',	5.2890,	7.1400,	1.0160,	12.6500,],
    ['2007-12-01',	5.2740,	6.9200,	1.0180,	12.6500,],
    ['2008-01-01',	4.4930,	6.3900,	1.0220,	12.6000,],
    ['2008-02-01',	5.0390,	6.8900,	1.0230,	12.5000,],
    ['2008-03-01',	5.1130,	7.5600,	1.0210,	12.5000,],
    ['2008-04-01',	5.1420,	7.4200,	1.0170,	12.4000,],
    ['2008-05-01',	4.7240,	6.9600,	1.0150,	12.4500,],
    ['2008-06-01',	4.6590,	6.6100,	1.0150,	12.4500,],
    ['2008-07-01',	4.7860,	6.5300,	1.0170,	12.5000,],
    ['2008-08-01',	5.1710,	7.4800,	1.0150,	12.6000,],
    ['2008-09-01',	6.2210,	8.6500,	1.0100,	12.6000,],
    ['2008-10-01',	6.7280,	12.8700,	1.0010,	12.7000,],
    ['2008-11-01',	7.6990,	20.1600,	0.9950,	12.8000,],
    ['2008-12-01',	8.0400,	22.8100,	0.9930,	12.8000,],
    ['2009-01-01',	7.6810,	25.3700,	0.9930,	12.9000,],
    ['2009-02-01',	8.2940,	25.0600,	0.9910,	14.2000,],
    ['2009-03-01',	7.7860,	21.2600,	0.9880,	14.3500,],
    ['2009-04-01',	8.1900,	18.4900,	0.9840,	14.4400,],
    ['2009-05-01',	7.5390,	15.1500,	0.9830,	14.5600,],
    ['2009-06-01',	7.8330,	13.3600,	0.9850,	14.5800,],
    ['2009-07-01',	7.6910,	13.0100,	0.9890,	14.6100,],
    ['2009-08-01',	8.1450,	12.6200,	0.9920,	14.6400,],
    ['2009-09-01',	8.1460,	12.2300,	0.9930,	14.6200,],
    ['2009-10-01',	7.1040,	9.9700,	0.9930,	14.5900,],
    ['2009-11-01',	6.8170,	8.6100,	0.9950,	14.5300,],
    ['2009-12-01',	6.1870,	7.9400,	0.9980,	14.4500,],
    ['2010-01-01',	5.5210,	6.6100,	1.0020,	14.3200,],
    ['2010-02-01',	5.3760,	6.1700,	1.0040,	13.9300,],
    ['2010-03-01',	4.6620,	5.0700,	1.0040,	13.7400,],
    ['2010-04-01',	4.1070,	4.6800,	1.0020,	13.5800,],
    ['2010-05-01',	4.0090,	4.6600,	1.0010,	13.5200,],
    ['2010-06-01',	3.2960,	4.4400,	1.0010,	13.4900,],
    ['2010-07-01',	3.2030,	4.3500,	1.0010,	13.4500,],
    ['2010-08-01',	3.5180,	4.2800,	1.0010,	13.3900,],
    ['2010-09-01',	3.5650,	4.2500,	1.0020,	13.3800,],
    ['2010-10-01',	3.4030,	4.2300,	1.0030,	13.3500,],
    ['2010-11-01',	3.8050,	4.2500,	1.0030,	13.2700,],
    ['2010-12-01',	4.2560,	4.3200,	1.0040,	13.1800,],
    ['2011-01-01',	4.0880,	4.3600,	1.0040,	13.0500,],
    ['2011-02-01',	3.8910,	4.3400,	1.0040,	12.6300,],
    ['2011-03-01',	3.8140,	4.3100,	1.0040,	12.5000,],
    ['2011-04-01',	3.4250,	4.1800,	1.0040,	12.4400,],
    ['2011-05-01',	3.5170,	4.2800,	1.0040,	12.3300,],
    ['2011-06-01',	3.6940,	4.3700,	1.0030,	12.2900,],
    ['2011-07-01',	3.6620,	4.4100,	1.0030,	12.2500,],
    ['2011-08-01',	3.8240,	4.8100,	1.0030,	12.1800,],
    ['2011-09-01',	4.4900,	5.6300,	1.0040,	12.1400,],
    ['2011-10-01',	5.4540,	6.8300,	1.0060,	12.0600,],
    ['2011-11-01',	5.4560,	7.0300,	1.0080,	12.0200,],
    ['2011-12-01',	5.6090,	7.2700,	1.0090,	11.9500,],
    ['2012-01-01',	5.5010,	7.1900,	1.0110,	11.9000,],
    ['2012-02-01',	5.4120,	7.0800,	1.0110,	11.8300,],
    ['2012-03-01',	5.6770,	7.0400,	1.0110,	11.8900,],
    ['2012-04-01',	5.8690,	7.0100,	1.0100,	11.9700,],
    ['2012-05-01',	6.0460,	7.1700,	1.0090,	12.0100,],
    ['2012-06-01',	6.1390,	7.3300,	1.0090,	12.0500,],
    ['2012-07-01',	5.5450,	7.4200,	1.0090,	12.0800,],
    ['2012-08-01',	5.3410,	7.3800,	1.0090,	12.1100,],
    ['2012-09-01',	5.5940,	7.3400,	1.0090,	12.1300,],
    ['2012-10-01',	5.7140,	7.3700,	1.0100,	12.1600,],
    ['2012-11-01',	5.7500,	7.5900,	1.0090,	12.1900,],
    ['2012-12-01',	5.8450,	7.6600,	1.0080,	12.2400,],
    ['2013-01-01',	5.4820,	7.5100,	1.0070,	12.2900,],
    ['2013-02-01',	4.7820,	7.3100,	1.0050,	12.6900,],
    ['2013-03-01',	5.1920,	7.3000,	1.0050,	12.7700,],
    ['2013-04-01',	5.1620,	7.4400,	1.0040,	12.8200,],
    ['2013-05-01',	5.4660,	7.3600,	1.0030,	12.7600,],
    ['2013-06-01',	5.8540,	7.2900,	1.0030,	12.7400,],
    ['2013-07-01',	5.4570,	7.2200,	1.0020,	12.7200,],
    ['2013-08-01',	5.5450,	7.1200,	1.0010,	12.6700,],
    ['2013-09-01',	5.6670,	7.1200,	1.0010,	12.6300,],
    ['2013-10-01',	5.6220,	7.0300,	1.0010,	12.6000,],
    ['2013-11-01',	5.8110,	7.0600,	1.0010,	12.5600,],
    ['2013-12-01',	5.9050,	7.2200,	1.0020,	12.4800,],
    ['2014-01-01',	5.8700,	7.2350,	1.0040,	12.5100,],
    ['2014-02-01',	6.0890,	7.2900,	1.0040,	12.3000,],
    ['2014-03-01',	6.9840,	9.1220,	1.0040,	12.3000,],
    ['2014-04-01',	7.3370,	9.4160,	1.0040,	12.1800,],
    ['2014-05-01',	7.3860,	9.9290,	1.0040,	12.2200,],
    ['2014-06-01',	7.3750,	9.7910,	1.0040,	12.2200,],
    ['2014-07-01',	7.6800,	9.8250,	1.0030,	12.2200,],
    ['2014-08-01',	7.7340,	10.3110,	1.0030,	12.2300,],
    ['2014-09-01',	7.8430,	10.4410,	1.0030,	12.2400,],
    ['2014-10-01',	7.9510,	10.8600,	1.0060,	12.2700,],
    ['2014-11-01',	8.9680,	12.3820,	1.0060,	12.3400,],
    ['2014-12-01',	12.1550,	20.5530,	1.0060,	12.3700,],
    ['2015-01-01',	13.9620,	22.2930,	1.0030,	12.4700,],
    ['2015-02-01',	12.1130,	17.7070,	1.0030,	14.1600,],
    ['2015-03-01',	11.9100,	16.4120,	1.0030,	14.4600,],
    ['2015-04-01',	11.0640,	14.8830,	0.9980,	14.5400,],
    ['2015-05-01',	9.7380,	13.6980,	0.9980,	14.4100,],
    ['2015-06-01',	9.8280,	12.9440,	0.9980,	14.2300,],
    ['2015-07-01',	9.0970,	12.3990,	0.9940,	14.0500,],
    ['2015-08-01',	9.2130,	12.0290,	0.9940,	13.9500,],
    ['2015-09-01',	9.3290,	11.9810,	0.9940,	13.7900,],
    ['2015-10-01',	9.5930,	11.9260,	0.9940,	13.7000,],
    ['2015-11-01',	9.8770,	11.8710,	0.9940,	13.5900,],
    ['2015-12-01',	10.0070,	11.8710,	0.9940,	13.4400,],
    ['2016-01-01',	9.3310,	12.0390,	0.9980,	13.3600,],
    ['2016-02-01',	8.9020,	12.0930,	0.9980,	12.5100,],
    ['2016-03-01',	9.2820,	12.0200,	0.9980,	12.2400,],
    ['2016-04-01',	9.6680,	11.8510,	0.9970,	12.4800,],
    ['2016-05-01',	9.4760,	11.5170,	0.9970,	12.6000,],
    ['2016-06-01',	9.9620,	11.2620,	0.9970,	12.7000,],
    ['2016-07-01',	9.7030,	11.0140,	0.9970,	12.7400,],
    ['2016-08-01',	9.3080,	10.9000,	0.9970,	12.7500,],
    ['2016-09-01',	9.2810,	10.5900,	0.9970,	12.7400,],
    ['2016-10-01',	9.3790,	10.5590,	0.9980,	12.7100,],
    ['2016-11-01',	9.1410,	10.5720,	0.9980,	12.6800,],
    ['2016-12-01',	8.5140,	10.5920,	0.9980,	12.6200,],
    ['2017-01-01',	8.3410,	10.5310,	0.9990,	12.4900,],
    ['2017-02-01',	9.1800,	10.4860,	0.9990,	11.8400,],
    ['2017-03-01',	9.4030,	10.3320,	0.9990,	11.9000,],
    ['2017-04-01',	8.8520,	9.9440,	0.9980,	11.8100,],
    ['2017-05-01',	8.3650,	9.4690,	0.9980,	11.6900,],
    ['2017-06-01',	7.9590,	9.1940,	0.9980,	11.6100,],
    ['2017-07-01',	7.8160,	9.1030,	0.9970,	11.5000,],
    ['2017-08-01',	7.6720,	9.0280,	0.9970,	11.4100,],
    ['2017-09-01',	7.4800,	8.7990,	0.9970,	11.2800,],
    ['2017-10-01',	7.4390,	8.5610,	1.0000,	11.1000,],
    ['2017-11-01',	7.2100,	8.3440,	1.0000,	10.9400,],
    ['2017-12-01',	6.7440,	8.1120,	1.0000,	10.7800,],
    ['2018-01-01',	6.3250,	7.7560,	1.0030,	10.6400,],
    ['2018-02-01',	6.2070,	7.4620,	1.0030,	9.8500,],
    ['2018-03-01',	5.8910,	7.3520,	1.0030,	9.8000,],
    ['2018-04-01',	6.2510,	7.3480,	1.0030,	9.7400,],
    ['2018-05-01',	6.2360,	7.3580,	1.0030,	9.6900,],
    ['2018-06-01',	6.4190,	7.3800,	1.0030,	9.6600,],
    ['2018-07-01',	6.4810,	7.4050,	1.0020,	9.6300,],
    ['2018-08-01',	6.7690,	7.5910,	1.0020,	9.6200,],
    ['2018-09-01',	6.8440,	7.9550,	1.0020,	9.5900,],
    ['2018-10-01',	6.8690,	8.3520,	1.0060,	9.5700,],
    ['2018-11-01',	7.0500,	8.4610,	1.0060,	9.5500,],
    ['2018-12-01',	7.1390,	8.6100,	1.0060,	9.5400,],
    ['2019-01-01',	7.1050,	8.6860,	1.0040,	9.5600,],
    ['2019-02-01',	7.3090,	8.6860,	1.0040,	9.8800,],
    ['2019-03-01',	6.9160,	8.5650,	1.0040,	10.1500,],
    ['2019-04-01',	7.2480,	8.4450,	1.0040,	10.4100,],
    ['2019-05-01',	6.8810,	8.3340,	1.0040,	10.5500,],
    ['2019-06-01',	6.8994,	8.1926,	1.0040,	10.5300,],
    ['2019-07-01',	6.6548,	7.9591,	1.0015,	10.2900,],
    ['2019-08-01',	6.5199,	7.3945,	1.0015,	10.2400,],
],
	columns=[
		u'Date0',
		names.CFN.SPT,
		names.CFN.MS6,
		names.CFN.HPR,
		names.CFN.MIR
	]
)
data[u'Date'] = data[u'Date0'].values.astype('datetime64[M]')

data.set_index(u'Date', inplace=True)
data.sort_index(inplace=True)

# --- fill nan values ---
data[names.CFN.SPT] = data[names.CFN.SPT].interpolate()
data[names.CFN.MS6] = data[names.CFN.MS6].interpolate()
data[names.CFN.HPR] = data[names.CFN.HPR].interpolate()
data[names.CFN.MIR] = data[names.CFN.MIR].interpolate()


def sql_macrohist():

    df = data.copy(deep=True)

    df.sort_index(inplace=True)
    df.reset_index(inplace=True)
    df[names.CFN.DAT] = pandas.to_datetime(df[u'Date'])
    del df[u'Date0']
    del df[u'Date']

    return df[[names.CFN.DAT,names.CFN.SPT,names.CFN.MS6,names.CFN.HPR,names.CFN.MIR]]