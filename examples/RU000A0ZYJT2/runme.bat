cd C:\CFM_Public

rem --- create json-file with inputs ---
python pricing.py --isin=RU000A0ZYJT2 --curdate=2019-04-15 --spread=120 -p="C:\CFM_Public\examples\RU000A0ZYJT2\run_params.json"  -i="C:\CFM_Public\examples\RU000A0ZYJT2\LLD 2019-04-01.csv"

rem --- run model ---
python runner.py -p="C:\CFM_Public\examples\RU000A0ZYJT2\run_params.json" -m=NumberOfMonteCarloScenarios=10 -m=RandomNumberSeed=123456 --dumpdir="C:\CFM_Public\examples\RU000A0ZYJT2"


pause