from parser import parser
from solver import Solver
import warnings
warnings.filterwarnings("ignore")


def main():
    args = parser()
    solver = Solver(args)
    solver.train()


if __name__ == '__main__':
    main()