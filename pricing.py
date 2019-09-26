# -*- coding: utf8 -*-
import json
import argparse

from impdata import gen_utility_json


def main(args):

    objrepr = json.dumps(gen_utility_json(args), indent=4, ensure_ascii=False, default=str).encode(args.encoding)
    if args.params is not None:
        with open(args.params, 'w') as f:
            f.write(objrepr)
    else:
        print objrepr


parser = argparse.ArgumentParser()
parser.add_argument("-p", "--params", action="store", type=str, help="result path to json-file with model params")
parser.add_argument("-i", "--inputs", action="store", type=str, help="path lld datafile")

parser.add_argument("--isin", action="store", type=str, help="Bond ISIN")
parser.add_argument("--curdate", action="store", type=str, help="Current Date")

parser.add_argument("--spread", action="store", type=float, help="Bond ID")
parser.add_argument("--price", action="store", type=float, help="Bond ID")
parser.add_argument("--scenarios", action="store", type=int, default=5000, help="NumberOfMonteCarloScenarios")
parser.add_argument("--horizon", action="store", type=int, default=12, help="ModelingHorizon")
parser.add_argument("--events", action="store", type=int, default=1000, help="NumberOfEventSimulations")
parser.add_argument("--clusters", action="store", default=10, type=int, help="NumberOfSyntheticLoans")
parser.add_argument("--seed", action="store", type=int, help="RandomNumberSeed")
parser.add_argument("--noconv", action="count", help="CalculationOfEffectiveIndicators == 0")

parser.add_argument("-e", "--encoding", action="store", type=str, default='utf8')


if __name__ == "__main__":

    main(parser.parse_args())

    # # === EXAMPLE OF USAGE FROM ANOTHER SCRIPT ===
    # argslist = [
    #     '--isin', 'RU000A0ZYL89',
    #     '--params', r'C:\CFM_Public\examples\RU000A0ZYL89\run_params.json',
    #     '--inputs', r'C:\CFM_Public\examples\RU000A0ZYL89\LLD 2019-04-01.csv',
    #     '--spread', '120',
    #     '--curdate', '2019-04-15',
    #     '--scenarios', '1',
    #     '--events', '1000',
    #     '--clusters', '10',
    #     '--seed', '123456',
    # ]
    # main(parser.parse_args(argslist))
    # # --- to use script from IDE ---

